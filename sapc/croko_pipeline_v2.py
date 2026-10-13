#!/usr/bin/env python3
"""Croko-Robust -> streaming Icefall Zipformer. Data stays on D:. Read --help.
Subcommands: download, manifest, patch.
"""
import argparse
import concurrent.futures
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HANDLE = 'sanjidh090/croko-robust'
HOME = Path.home()
DATA = HOME / 'croko_training' / 'data'
WORK = Path('/mnt/d/Croko_Training')
DEV = Path('/mnt/g/SAPC2_Dataset/Dev/dev_metadata_cleaned.csv')
BPE = Path('/mnt/g/SAPC2_Dataset/Pretrained_Zipformer/model/data/lang_bpe_510/bpe.model')
RECIPE = HOME / 'icefall/egs/librispeech/ASR/zipformer'


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s or '').casefold()).strip()


def train_text(s):
    return re.sub(r'\s+', ' ', re.sub(r"[^A-Z0-9' ]+", ' ', str(s).upper())).strip()


def selected_jobs(limit=0):
    import pandas as pd
    for f in [DATA / 'completed.jsonl', DEV]:
        if not f.is_file():
            raise FileNotFoundError(f'Missing {f}')
    dev = pd.read_csv(DEV, usecols=['transcript'], dtype=str)
    protected = {norm(x) for x in dev['transcript'].dropna()}
    protected.discard('')
    with (DATA / 'completed.jsonl').open(encoding='utf-8') as f:
        jobs = [json.loads(x) for x in f if x.strip()]
    filtered, reasons = [], Counter()
    for j in jobs:
        t = norm(j.get('text'))
        if not t:
            reasons['empty'] += 1
            continue
        if t in protected:
            reasons['dev_overlap'] += 1
            continue
        if not j.get('shard') or not j.get('audio_member'):
            reasons['missing_audio_path'] += 1
            continue
        filtered.append(j)
    if limit and limit < len(filtered):
        # Balanced across the four recorded TTS voices, reproducible.
        by_voice = defaultdict(list)
        for j in filtered:
            by_voice[j.get('voice', 'unknown')].append(j)
        rng = random.Random(42)
        for bucket in by_voice.values():
            rng.shuffle(bucket)
        selected = []
        names = sorted(by_voice)
        i = 0
        while len(selected) < limit:
            voice = names[i % len(names)]
            if by_voice[voice]:
                selected.append(by_voice[voice].pop())
            i += 1
        filtered = selected
    print(f'Selected {len(filtered)} clean jobs; skipped {dict(reasons)}', flush=True)
    return filtered


def relative_audio(j, augmented=False):
    # Kaggle lists WAVs as individual files inside directories named after the .tar stem.
    folder = Path(str(j['shard'])).stem
    member = j['audio_member']
    if augmented:
        variants = j.get('variants') or []
        for v in variants:
            if v.get('variant') == 'augmented' and (v.get('augmentation') or {}).get('preserves_transcript') is True and v.get('audio_member'):
                return Path('tts_robust_output') / folder / v['audio_member']
        return None
    return Path('tts_robust_output') / folder / member


def ensure_download(task, retries=3):
    rel, output_root = task
    out = output_root / rel
    if out.is_file() and out.stat().st_size > 44:
        return ('present', str(rel), '')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_parent = output_root / '.kaggle_tmp'
    tmp_parent.mkdir(parents=True, exist_ok=True)
    last = ''
    for trial in range(retries):
        # Kaggle CLI sometimes flattens its --unzip output. Isolate every
        # individual download, then move the file to our expected path.
        with tempfile.TemporaryDirectory(prefix='croko_', dir=tmp_parent) as tmp:
            temp = Path(tmp)
            cmd = ['kaggle', 'datasets', 'download', HANDLE,
                   '-f', rel.as_posix(), '-p', str(temp), '--unzip', '-q']
            r = subprocess.run(cmd, capture_output=True, text=True)
            matches = [f for f in temp.rglob(rel.name) if f.is_file() and f.stat().st_size > 44]
            if r.returncode == 0 and matches:
                os.replace(matches[0], out)
                return ('downloaded', str(rel), '')
            last = (r.stderr or r.stdout or f'exit={r.returncode}, file not found after download')[-350:]
        time.sleep(1.2 * (trial + 1))
    return ('failed', str(rel), last)

def do_download(args):
    if not shutil.which('kaggle'):
        sys.exit('Install Kaggle CLI: pip install -U kaggle')
    jobs = selected_jobs(args.limit)
    tasks = []
    for j in jobs:
        tasks.append((relative_audio(j), args.work / 'audio'))
        if args.include_aug:
            rel = relative_audio(j, augmented=True)
            if rel is not None:
                tasks.append((rel, args.work / 'audio'))
    # Deduplicate paths; preserve processing order.
    tasks = list(dict.fromkeys(tasks))
    print(f'Files to check/download: {len(tasks)}; destination: {args.work / "audio"}', flush=True)
    tally = Counter()
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (status, path, err) in enumerate(pool.map(ensure_download, tasks), 1):
            tally[status] += 1
            if status == 'failed':
                errors.append({'file': path, 'error': err})
            if i % 100 == 0 or i == len(tasks):
                print(f'{i}/{len(tasks)} | {dict(tally)}', flush=True)
    audit_dir = args.work / 'logs'
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / 'download_failures.json').write_text(json.dumps(errors, indent=2))
    print('DOWNLOAD COMPLETE:', dict(tally))
    if errors:
        print('Failed files recorded:', audit_dir / 'download_failures.json')


def do_manifest(args):
    try:
        import soundfile as sf
        import sentencepiece as spm
        from lhotse import CutSet, MonoCut, Recording, SupervisionSegment
    except ImportError as e:
        sys.exit(f'Missing dependency: {e}. Run: pip install soundfile sentencepiece lhotse')
    if not BPE.is_file():
        sys.exit(f'BPE model not found: {BPE}')
    sp = spm.SentencePieceProcessor(model_file=str(BPE))
    jobs = selected_jobs(args.limit)
    cuts = []
    reasons = Counter()
    voices = Counter()
    manifest_dir = args.work / 'manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for j in jobs:
        text = train_text(j['text'])
        if not text:
            reasons['empty_normalized'] += 1
            continue
        tokens = sp.encode(text, out_type=int)
        if not tokens or sp.unk_id() in tokens:
            reasons['tokenizer_unk'] += 1
            continue
        for is_aug in [False, True] if args.include_aug else [False]:
            rel = relative_audio(j, augmented=is_aug)
            if rel is None:
                continue
            path = args.work / 'audio' / rel
            if not path.is_file():
                reasons['not_downloaded'] += 1
                continue
            try:
                info = sf.info(str(path))
            except Exception:
                reasons['unreadable_audio'] += 1
                continue
            duration = info.frames / info.samplerate if info.samplerate else 0
            if info.samplerate != 16000 or info.channels != 1:
                reasons['unexpected_audio_format'] += 1
                continue
            # Original Icefall recipe cuts outside this duration interval.
            if not (1.0 <= duration <= 20.0):
                reasons['duration_outside_1_20s'] += 1
                continue
            # Avoid short audio with too many BPE labels after subsampling.
            if len(tokens) >= max(1, int(duration * 25) - 2):
                reasons['too_many_tokens_for_audio'] += 1
                continue
            key = 'croko-' + j['id'] + ('-aug' if is_aug else '')
            try:
                rec = Recording.from_file(str(path), recording_id=key)
                sup = SupervisionSegment(id=key, recording_id=key, start=0.0,
                      duration=rec.duration, channel=0, text=text,
                      speaker=f"croko_{j.get('voice', 'unknown')}")
                cuts.append(MonoCut(id=key, start=0.0, duration=rec.duration,
                                     channel=0, recording=rec, supervisions=[sup]))
                voices[j.get('voice', 'unknown')] += 1
            except Exception as exc:
                reasons['lhotse_construction_error'] += 1
                if reasons['lhotse_construction_error'] < 4:
                    print('Manifest warning:', exc)
    if not cuts:
        sys.exit('NO VALID CUTS. Check WAV download paths and --limit / --include-aug flags.')
    target = manifest_dir / 'croko_cuts_train.jsonl.gz'
    CutSet.from_cuts(cuts).to_file(target)
    # Sanity read-back; avoids launching a run with broken manifest encoding.
    from lhotse import load_manifest_lazy
    check = next(iter(load_manifest_lazy(target)))
    print('MANIFEST:', target)
    print('Cuts:', len(cuts), 'voices:', dict(voices), 'excluded/missing:', dict(reasons))
    print('First cut:', check.id, 'duration:', check.duration, 'text:', check.supervisions[0].text[:90])
    (args.work / 'logs').mkdir(parents=True, exist_ok=True)
    (args.work / 'logs' / 'manifest_counts.json').write_text(json.dumps(
        {'cuts': len(cuts), 'voices': dict(voices), 'excluded': dict(reasons)}, indent=2))


def do_patch(args):
    src = args.recipe / 'finetune.py'
    dst = args.recipe / 'finetune_croko.py'
    if not src.is_file():
        sys.exit(f'Cannot find original recipe: {src}')
    if dst.exists() and not args.overwrite:
        sys.exit(f'{dst} exists; rerun with --overwrite to recreate it.')
    script = src.read_text()
    pat = re.compile(r'(?m)^[ \t]*train_cuts[ \t]*=[ \t]*load_manifest_lazy\([^\n]*\)[ \t]*$')
    matches = list(pat.finditer(script))
    if len(matches) != 1:
        sys.exit('Expected exactly one train_cuts=load_manifest_lazy(...) assignment; found '
                 + str(len(matches)) + '. Send: grep -n "train_cuts =" finetune.py')
    replacement = '''    from lhotse import CutSet
    real_cuts = load_manifest_lazy("/mnt/g/SAPC2_Dataset/manifests/sapc2_cuts_train_local.jsonl.gz")
    synthetic_cuts = load_manifest_lazy("/mnt/d/Croko_Training/manifests/croko_cuts_train.jsonl.gz")
    # Sample ~80% real / ~20% synthetic until synthetic is exhausted.
    # This is a short adaptation experiment, not an exhaustive sweep of real data.
    train_cuts = CutSet.mux(real_cuts, synthetic_cuts, weights=[0.8, 0.2], stop_early=True, seed=42)'''
    script = pat.sub(replacement, script, count=1)
    if not re.search(r'(?m)^\s*valid_cuts\s*=\s*load_manifest_lazy\(', script):
        sys.exit('Could not confirm real Dev validation in source finetune.py. Nothing written.')
    if 'def main(' not in script:
        sys.exit('Could not confirm full training script. Nothing written.')
    dst.write_text(script)
    result = subprocess.run([sys.executable, '-m', 'py_compile', str(dst)], capture_output=True, text=True)
    if result.returncode:
        dst.unlink(missing_ok=True)
        sys.exit(f'Patched recipe syntax error:\n{result.stderr}')
    print('Created:', dst)
    print('Original preserved:', src)
    print('Training mix: approx 80% real + 20% synthetic until one dataset exhausted.')
    for line in script.splitlines():
        if 'valid_cuts = load_manifest_lazy(' in line:
            print('Validation:', line.strip())


def cli():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    for name, func in [('download', do_download), ('manifest', do_manifest), ('patch', do_patch)]:
        p = sub.add_parser(name)
        p.set_defaults(func=func)
        p.add_argument('--work', type=Path, default=WORK,
                       help='D: working folder (/mnt/d/Croko_Training)')
        p.add_argument('--limit', type=int, default=0,
                       help='0=all eligible clean jobs; nonzero=balanced sample')
        p.add_argument('--include-aug', action='store_true',
                       help='Also download/include transcript-preserving augmented WAVs')
        if name == 'download':
            p.add_argument('--workers', type=int, default=6)
        if name == 'patch':
            p.add_argument('--recipe', type=Path, default=RECIPE)
            p.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()
    args.func(args)

if __name__ == '__main__':
    cli()
