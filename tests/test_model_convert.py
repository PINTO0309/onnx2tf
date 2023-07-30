import argparse
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import glob
import platform
import shutil
import sys
import tempfile
import onnx
import tensorflow as tf
import onnx2tf

_CFG = {}


class Results:
    """Tracks the detailed status and counts for the report."""

    def __init__(self):
        self.details = []
        self.model_count = 0
        self.total_count = 0
        self.pass_count = 0
        self.warn_count = 0
        self.fail_count = 0
        self.skip_count = 0

    def append_detail(self, line):
        """Append a line of detailed status."""
        self.details.append(line)

    @classmethod
    def _report(cls, line):
        if _CFG['verbose']:
            print(line)
        if not _CFG['dry_run']:
            with open(_CFG['report_filename'], 'a') as file:
                file.write(line + '\n')

    def generate_report(self):
        """Generate the report file."""
        if _CFG['verbose']:
            print(f'Writing {_CFG["report_filename"]}{" (dry_run)" if _CFG["dry_run"] else ""}\n')
        self._report(f'*Report generated at {datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}{_CFG["github_actions_md"]}.*')

        self._report('\n## Environment')
        self._report('Package | Version')
        self._report('---- | -----')
        self._report(f'Platform | {platform.platform()}')
        sys_version = sys.version.replace("\n", " ")
        self._report(f'Python | {sys_version}')
        self._report(f'onnx | {onnx.__version__}')
        self._report(f'onnx2tf | {_CFG["onnx2tf_version_md"]}')
        self._report(f'tensorflow | {tf.__version__}')

        self._report('\n## Summary')
        self._report(f'Value | Count')
        self._report(f'---- | -----')
        self._report(f'Total | {self.total_count}')
        self._report(f':heavy_check_mark: Passed | {self.pass_count}')
        self._report(f':warning: Limitation | {self.warn_count}')
        self._report(f':x: Failed | {self.fail_count}')
        self._report(f':heavy_minus_sign: Skipped | {self.skip_count}')

        self._report('\n## Details')
        self._report('\n'.join(self.details))
        self._report('')

    def summary(self):
        """Return the report summary (counts, report location) as a string."""
        return (
            f'Total: {self.total_count}, '+
            f'Passed: {self.pass_count}, '+
            f'Limitation: {self.warn_count}, '+
            f'Failed: {self.fail_count}, '+
            f'Skipped: {self.skip_count}\n'+
            f'Report: {_CFG["report_filename"]}{" (dry_run)" if _CFG["dry_run"] else ""}'
        )


def _del_location(loc):
    if not _CFG['dry_run'] and os.path.exists(loc):
        if os.path.isdir(loc):
            shutil.rmtree(loc)
        else:
            os.remove(loc)


def _report_check_model(model):
    """Use ONNX checker to test if model is valid and return a report string."""
    try:
        onnx.checker.check_model(model)
        return ''
    except Exception as ex:
        _del_location(_CFG['untar_directory'])
        first_line = str(ex).strip().split('\n')[0].strip()
        return f'{type(ex).__name__}: {first_line}'


def _report_convert_model(file_path):
    """Test conversion and returns a report string."""
    try:
        onnx2tf.convert(
            input_onnx_file_path=file_path,
            output_folder_path=_CFG['output_directory'],
            output_nms_with_dynamic_tensor=True,
            disable_strict_mode=True,
            disable_model_save=True,
            verbosity="error",
        )
        os.remove(file_path)
        for tflitepath in glob.glob(f"{_CFG['output_directory']}/*.tflite"):
            os.remove(tflitepath)
        return ''
    except Exception as ex:
        _del_location(_CFG['untar_directory'])
        _del_location(_CFG['output_directory'])
        stack_trace = str(ex).strip().split('\n')
        if len(stack_trace) > 1:
            err_msg = stack_trace[-1].strip()
            # OpUnsupportedException gets raised as a RuntimeError
            if 'OP is not yet implemented' in str(ex):
                err_msg = err_msg.replace(type(ex).__name__, 'OpUnsupportedException')
            return err_msg
        return f'{type(ex).__name__}: {stack_trace[0].strip()}'


def _report_model(file_path, results=Results(), onnx_model_count=1):
    """Generate a report status for a single model, and append it to results."""
    if _CFG['dry_run']:
        ir_version = ''
        opset_version = ''
        check_err = ''
        convert_err = ''
        emoji_validated = ''
        emoji_converted = ''
        emoji_overall = ':heavy_minus_sign:'
        results.skip_count += 1
    else:
        if _CFG['verbose']:
            print('Testing', file_path)
        model = onnx.load(file_path)
        ir_version = model.ir_version
        opset_version = model.opset_import[0].version
        check_err = _report_check_model(model)
        del model
        convert_err = '' if check_err else _report_convert_model(file_path)

        if not check_err and not convert_err:
            # ran successfully
            emoji_validated = ':ok:'
            emoji_converted = ':ok:'
            emoji_overall = ':heavy_check_mark:'
            results.pass_count += 1
        elif not check_err and convert_err:
            # validation pass, but conversion did not
            emoji_validated = ':ok:'
            emoji_converted = convert_err
            emoji_overall = ':x:'
            results.fail_count += 1
        elif check_err and not convert_err:
            # validation did not, but conversion pass
            emoji_validated = check_err
            emoji_converted = ':ok:'
            emoji_overall = ':heavy_check_mark:'
            results.pass_count += 1
        else:
            # validation failed
            emoji_validated = check_err
            emoji_converted = ':heavy_minus_sign:'
            emoji_overall = ':x:'
            results.fail_count += 1

        results.append_detail(
            f'{emoji_overall} | {onnx_model_count}. {os.path.splitext(os.path.basename(file_path))[0]} | '+
            f'{ir_version} | {opset_version} | {emoji_validated} | {emoji_converted}')


def _configure(
    models_dir='models',
    output_dir=tempfile.gettempdir(),
    verbose=False,
    dry_run=False,
):
    """Validate the configuration."""
    if not os.path.isdir(models_dir):
        os.makedirs(models_dir, exist_ok=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    _CFG['models_dir'] = os.path.normpath(models_dir)
    _CFG['verbose'] = verbose
    _CFG['dry_run'] = dry_run

    _configure_env()

    norm_output_dir = os.path.normpath(output_dir)
    _CFG['untar_directory'] = os.path.join(norm_output_dir, 'test_model_and_data')
    _CFG['output_directory'] = os.path.join(norm_output_dir, 'test_model_pb')
    _CFG['report_filename'] = os.path.join(norm_output_dir, _CFG['report_filename'])


def _configure_env():
    """Set additional configuration based on environment variables."""
    ref = os.getenv('GITHUB_REF')
    repo = os.getenv('GITHUB_REPOSITORY')
    sha = os.getenv('GITHUB_SHA')
    run_id = os.getenv('GITHUB_RUN_ID')

    _CFG['report_filename'] = 'model_status.md'

    if repo:
        # actions ([run_id](url))
        actions_url = f'https://github.com/{repo}/actions'
        _CFG['github_actions_md'] = f' via [GitHub Actions]({actions_url})'
        if run_id:
            run_link = f' ([{run_id}]({actions_url}/runs/{run_id}))'
            _CFG['github_actions_md'] += run_link
    else:
        _CFG['github_actions_md'] = ''

    _CFG['onnx2tf_version_md'] = onnx2tf.__version__
    if sha and repo:
        # version ([sha](url))
        commit_url = f'https://github.com/{repo}/commit/{sha}'
        _CFG['onnx2tf_version_md'] += f' ([{sha[0:7]}]({commit_url}))'


def model_convert_report(
    models_dir='models',
    output_dir=tempfile.gettempdir(),
    verbose=False,
    dry_run=False,
):
    """model_convert_report.

    Parameters
    ----------
    models_dir: str
        directory that contains ONNX models
    output_dir: str
        directory for the generated report and converted model
    verbose: bool
        verbose output
    dry_run: bool
        process directory without doing conversion

    Returns
    ----------
    report: str
        Results object containing detailed status and counts for the report.
    """

    _configure(models_dir, output_dir, verbose, dry_run)
    _del_location(_CFG['report_filename'])
    _del_location(_CFG['output_directory'])
    _del_location(_CFG['untar_directory'])

    # run tests first, but append to report after summary
    results = Results()
    for root, subdir, files in os.walk(_CFG['models_dir']):
        subdir.sort()
        results.model_count += 1
        results.append_detail('')
        results.append_detail(f'### {results.model_count}. {os.path.basename(root)}')
        results.append_detail('')
        results.append_detail(
            'Status | Model | IR | Opset | ONNX Checker | onnx2tf Converted'
        )
        results.append_detail(
            '------ | ----- | -- | ----- | ------------ | -----------------'
        )
        onnx_model_count = 0
        file_path = ''
        for item in sorted(files):
            if item.endswith('.onnx'):
                file_path = f'{_CFG["models_dir"]}/{item}'
                onnx_model_count += 1
                results.total_count += 1
                _report_model(file_path, results, onnx_model_count)
    return results


if __name__ == '__main__':
    tempdir = tempfile.gettempdir()
    parser = argparse.ArgumentParser(description='Test converting ONNX models to TensorFlow.')
    parser.add_argument(
        '-m',
        '--models',
        default='models',
        help='ONNX model directory (default: models)'
    )
    parser.add_argument(
        '-o',
        '--output',
        default=tempdir,
        help=f'output directory (default: {tempdir})'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='verbose output'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='process directory without doing conversion'
    )
    args = parser.parse_args()
    report = model_convert_report(
        args.models,
        args.output,
        args.verbose,
        args.dry_run,
    )
    report.generate_report()
    print(report.summary())
