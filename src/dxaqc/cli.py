"""CLI for the offline DXA quality-control application."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .api import analyze, create_predictor, default_model_dir
from .report import write_report


def model_arguments(parser):
    parser.add_argument('--model-dir', type=Path, default=default_model_dir(), help='папка с manifest.json и всеми весами')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps', 'auto'], default='cpu')
    parser.add_argument('--threads', type=int, default=2, help='CPU threads, от 1 до 8; по умолчанию 2')


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == 'doctor':
            parser = argparse.ArgumentParser(prog='dxaqc doctor', description='Проверка полноты и SHA-256 весов без загрузки из сети')
            parser.add_argument('--model-dir', type=Path, default=default_model_dir())
            args = parser.parse_args(argv[1:])
            from .bundle import verify_bundle
            manifest = verify_bundle(args.model_dir)
            print(json.dumps({'status': 'ok', 'model_id': manifest['model_id'], 'files': len(manifest['files']), 'version': __version__}, ensure_ascii=False))
            return 0
        if argv and argv[0] == 'serve':
            parser = argparse.ArgumentParser(prog='dxaqc serve', description='Локальный HTTP API пакетной обработки')
            model_arguments(parser)
            parser.add_argument('--host', default='127.0.0.1')
            parser.add_argument('--port', type=int, default=8080)
            args = parser.parse_args(argv[1:])
            from .server import serve
            serve(create_predictor(args.model_dir, device=args.device, threads=args.threads), args.host, args.port)
            return 0
        parser = argparse.ArgumentParser(prog='dxaqc', description='Офлайн контроль качества DXA: DICOM / папка / ZIP → CSV/XLSX')
        parser.add_argument('--version', action='version', version=__version__)
        parser.add_argument('--input', '-i', type=Path, required=True)
        parser.add_argument('--output', '-o', type=Path, required=True)
        model_arguments(parser)
        parser.add_argument('--path-mode', choices=['file', 'dir'], default='file')
        parser.add_argument('--details', action='store_true', help='добавить error_message')
        parser.add_argument('--strict-columns', action='store_true', help='8 колонок ТЗ и quality_prob (по умолчанию)')
        parser.add_argument('--scores', action='store_true', help='добавить непрерывные оценки для валидации')
        parser.add_argument('--type-probs', action='store_true',
                            help='добавить prob_<код>: вероятность каждого типа нарушения, 0.5 = порог этой причины')
        parser.add_argument('--fail-on-error', action='store_true', help='код 3 при хотя бы одном Failure; отчёт сохраняется')
        parser.add_argument('--no-view-gate', action='store_true',
                            help='не отклонять снимки вне области задачи (предплечье, боковая проекция)')
        parser.add_argument('--overlays', type=Path, help='папка или .zip: визуализация ориентиров (PNG + DICOM Secondary Capture)')
        parser.add_argument('--sr', type=Path, help='папка или .zip: DICOM SR — текстовый отчёт с вердиктом и измерениями на каждый снимок')
        parser.add_argument('--log', type=Path)
        parser.add_argument('--predictor', choices=['release', 'dummy', 'spine_rules', 'hip_embed_lr', 'mvp', 'hip_cnn'], default='release', help='legacy-модели только для воспроизведения экспериментов')
        parser.add_argument('-v', '--verbose', action='store_true')
        args = parser.parse_args(argv)
        if args.output.suffix.lower() not in ('.csv', '.xlsx'):
            raise ValueError('Выходной файл должен иметь расширение .csv или .xlsx')
        if args.strict_columns and args.scores:
            raise ValueError('--strict-columns несовместим с --scores')
        if not args.input.exists():
            raise FileNotFoundError(f'Входной путь не существует: {args.input}')
        log_path = args.log or args.output.with_suffix('.log')
        summary_path = args.output.with_suffix('.summary.json')
        output_paths = [args.output, log_path, summary_path]
        if len({p.resolve() for p in output_paths}) != 3 or args.input.resolve() in {p.resolve() for p in output_paths}:
            raise ValueError('Input, output, log and summary paths must not collide')
        # Output/log names may themselves refer to a DICOM inside the input folder.
        # Inspect before opening FileHandler or excluding destinations from discovery.
        import pydicom
        for destination in output_paths:
            if not destination.is_file():
                continue
            try:
                header = pydicom.dcmread(destination, force=True, stop_before_pixels=True,
                                        specific_tags=['Rows', 'Columns', 'SOPClassUID', 'SOPInstanceUID'])
            except Exception:
                header = None
            if header and any(key in header for key in ('Rows', 'Columns', 'SOPClassUID', 'SOPInstanceUID')):
                raise ValueError(f'Refusing to overwrite a DICOM input: {destination}')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                            format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                            handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler(sys.stderr)], force=True)
        logging.getLogger('pydicom').setLevel(logging.ERROR)
        predictor = None
        if args.predictor != 'release':
            logging.warning('Experimental predictor=%s; not the release model', args.predictor)
            # The runtime image ships only what the release model needs; the legacy predictors import
            # scikit-learn / scipy lazily (inside predict), so check up front instead of failing every file.
            import importlib.util
            missing = [m for m in ('sklearn', 'scipy', 'joblib') if importlib.util.find_spec(m) is None]
            if args.predictor != 'dummy' and missing:
                raise ValueError(f'Предиктор {args.predictor} экспериментальный и не входит в релизную сборку '
                                 f'(нет {", ".join(missing)}); используйте --predictor release '
                                 f'или установите зависимости разработки: pip install -e .')
            from .predictor import load_predictor
            predictor = load_predictor(args.predictor)
        import tempfile

        def out_target(value, name):
            """A folder, or a .zip filled from a temporary folder at the end."""
            if value is None:
                return None, None
            if value.resolve() == args.input.resolve() or args.input.resolve() in value.resolve().parents:
                raise ValueError(f'--{name} must be outside the input folder')
            if value.suffix.lower() == '.zip':
                return value, Path(tempfile.mkdtemp(prefix=f'dxaqc_{name}_'))
            return None, value

        overlay_zip, overlay_dir = out_target(args.overlays, 'overlays')
        sr_zip, sr_dir = out_target(args.sr, 'sr')
        if args.no_view_gate:
            predictor = predictor or create_predictor(args.model_dir, device=args.device, threads=args.threads)
            if hasattr(predictor, 'use_view_gate'):
                predictor.use_view_gate = False
                logging.warning('view gate disabled: out-of-scope images will get a verdict')
        rows, stats = analyze(args.input, predictor=predictor, model_dir=args.model_dir, device=args.device,
                              threads=args.threads, path_mode=args.path_mode, exclude=output_paths,
                              overlay_dir=overlay_dir, sr_dir=sr_dir)
        for zip_path, folder in ((overlay_zip, overlay_dir), (sr_zip, sr_dir)):
            if zip_path is None:
                continue
            import shutil
            import zipfile
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in sorted(folder.rglob('*')):
                    if f.is_file():
                        zf.write(f, f.relative_to(folder).as_posix())
            shutil.rmtree(folder, ignore_errors=True)
        write_report(rows, args.output, strict_columns=args.strict_columns or not args.details, scores=args.scores,
                     type_probs=args.type_probs)
        summary_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(stats, ensure_ascii=False))
        if not rows:
            return 4
        return 3 if args.fail_on_error and stats['failure'] else 0
    except KeyboardInterrupt:
        print('Обработка прервана пользователем', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'dxaqc: {type(error).__name__}: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
