__author__ = 'josep'

import sys
import gzip
import json
from retrying import retry
from boto.s3.connection import S3Connection
import boto
import boto.s3.key
import zlib
from logging.config import dictConfig
import os
import signal
import time
import subprocess
from inspect import getframeinfo, stack
import psutil
import pandas as pd
import multiprocessing as mp

if 'AIRFLOW_TMP_DIR' in os.environ.keys():
    pass

from Utilities import pandas_utils as p_ut

S3_BUCKET = 'airbnb-emr'
S3_CONTAINER = 'capacity_planning'

# TODO S3 read and write


class suppress_stdout_stderr(object):
    """
    from https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    Usage: with suppress_stdout_err():
               blah
    """
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]    # Open a pair of null files
        self.save_fds = (os.dup(1), os.dup(2))                                # Save the actual stdout (1) and stderr (2) file descriptors.

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)   # does not close!
        os.dup2(self.null_fds[1], 2)   # does not close!

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)   # does not close!
        os.dup2(self.save_fds[1], 2)   # does not close!

        # Close open fds
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])
        os.close(self.save_fds[0])
        os.close(self.save_fds[1])


def set_logger(h_dir, l_cfg_file, logger_dir, name, l_level):
    # set logger
    l_dir = h_dir + logger_dir if logger_dir != '/var/log' else logger_dir
    if not(os.path.isdir(l_dir)):
        os.makedirs(l_dir)
    ts = int(time.time())
    log_file = l_dir + '/' + name + '_' + str(ts)

    # read logger dict and adjust with cfg changes
    with open(h_dir + l_cfg_file, 'r') as fp:
        logger_dict = json.load(fp)
    logger_dict['handlers']['file']['filename'] = log_file
    logger_dict['handlers']['file']['level'] = l_level
    for h in logger_dict['loggers']:
        logger_dict['loggers'][h]['level'] = l_level
    dictConfig(logger_dict)
    print('Log file: ' + log_file)


def gb_memory_usage():   # return the memory usage in GB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(1024 * 1024 * 1024)
    return mem


def grep_proc(string):  # string is either a string to match or a list of strings to match. NO RX for now
    if string:
        my_list = list(string) if isinstance(string, list) else [string]
        out = dict()
        for f in os.popen('ps -efw'):
            for s in my_list:
                if s in f:
                    if s not in out:
                        out[s] = dict()
                    vf = f.split(' ')
                    out[s][vf[1]] = [vf[2], f]  # out[s][pid] = [ppid, ps output string]
        return out
    else:
        print('missing match strings')


def set_dirs(script_file):
    path = os.path.realpath(script_file)
    curr_dir = os.path.dirname(path)   # script directory
    return curr_dir


def to_gzip(f_name):
    if os.path.isfile(f_name):
        if f_name[-3:] != '.gz':
            os.system('gzip -f ' + f_name)
            return f_name + '.gz'
        return f_name
    return None


def read_json_lines(fname):
    f = gzip.open(fname, 'r') if fname[-3:] == '.gz' else open(fname, 'r')
    data_list = [json.loads(line) for line in f]
    f.close()
    return data_list


def df2json(df, f_out):
    """
    NOTE: only works if columns contain json-like elements: numbers, strings, lists, dict. Does not work for sets.
    write a df into a by-line json file
    df: input DF
    f_out: output file
    """
    f = gzip.open(f_out, 'wb') if f_out[-3:] == '.gz' else open(f_out, 'wb')
    f.write('\n'.join(r[1].to_json(orient='index') for r in df.iterrows()))
    f.close()


def df2json_gen(df, f_out):
    """
    NOTE: only works if columns contain json-like elements: numbers, strings, lists, dict. Does not work for sets.
    Generator version: slower but scales to larger DF's
    write a df into a by-line json file
    df: input DF
    f_out: output file
    """
    f = gzip.open(f_out, 'wb') if f_out[-3:] == '.gz' else open(f_out, 'wb')
    f.writelines((r[1].to_json(orient='index') + '\n' for r in df.iterrows()))
    f.close()


def df_to_json_lines(df, f_out):
    return df2json(df, f_out)


@retry(stop_max_attempt_number=5, wait_fixed=2000)
def read_from_s3(s3_file, local_file, s3_bucket='attune-data', s3_cfg='/home/josep/.s3cfg', aws_connection=None):
    """
    get the s3 file in s3_path and put it in local_path. No pattern matching
    :param s3_file: the full s3 path excluding the bucket
    :param local_file: full file local path
    :param s3_bucket: s3 bucket that contains the file to download
    :param s3_cfg: full path for the s3cmd cfg file containing access and secret key.
    :param aws_connection: object returned from S3Connection()
    :return: Nothing
    """
    if aws_connection is None:
        aws_connection = set_s3_connection(s3_cfg)
    bucket = aws_connection.get_bucket(s3_bucket)
    file_key = boto.s3.key.Key(bucket, s3_file)
    file_key.get_contents_to_filename(local_file)


def get_aws_keys(s3_cfg='/home/josep/.s3cfg'):
    access_key, secret_key = None, None
    with open(s3_cfg) as fp:
        lines = fp.readlines()
        for l in lines:
            if 'access_key' in l:
                lhs, access_key = [x.strip() for x in l.split(' = ')]
            if 'secret_key' in l:
                lhs, secret_key = [x.strip() for x in l.split(' = ')]
    return access_key, secret_key


def set_s3_connection(s3_cfg='/home/josep/.s3cfg'):
    access_key, secret_key = get_aws_keys(s3_cfg)
    return S3Connection(access_key, secret_key)


def write_to_s3(s3_file, local_file, s3_bucket='attune-data', s3_cfg='/home/josep/.s3cfg', aws_connection=None):
    if aws_connection is None:
        aws_connection = set_s3_connection(s3_cfg)
    bucket = aws_connection.get_bucket(s3_bucket)
    bucket_key = boto.s3.key.Key(bucket)
    bucket_key.key = s3_file
    bucket_key.set_contents_from_filename(local_file)


@retry(stop_max_attempt_number=5, wait_fixed=2000)
def stream_gzip_decompress(stream):
    dec = zlib.decompressobj(32 + zlib.MAX_WBITS)  # offset 32 to skip the header
    for chunk in stream:
        rv = dec.decompress(chunk)
        if rv:
            yield rv


def set_data_file_path(fname, is_airflow=False):  # no extension
    if is_airflow is True:
        # return 's3://' + S3_BUCKET + '/' + S3_CONTAINER + '/' + fname   # no extension
        return None
    else:
        return os.path.expanduser('~/my_tmp/' + fname)


def read_data_file(fpath, is_airflow=False):
    if is_airflow is True:
        return None
        # fname = os.path.basename(fpath)
        # ext = '.'.join(fpath.split('/')[-1].split('.')[1:])
        # local_filename = TempUtils.datafile(fname + '.' + ext)
        # s3 = S3Hook()
        # key = s3.get_key(fpath)
        # with open(local_filename, 'w') as local_file:
        #     my_print('reading file: ' + str(fpath))
        #     key.get_contents_to_file(local_file)
        # return read_data_file(local_filename, is_airflow=False)
    else:
        if os.path.isfile(fpath):
            my_print('reading file: ' + str(fpath))
            if '.par' in fpath:
                try:
                    return pd.read_parquet(fpath)
                except:
                    my_print('WARNING: corrupted? file  ' + str(fpath))
                    return None
            elif '.csv.gz' in fpath:
                return pd.read_csv(fpath)
            else:
                my_print('could not read ' + fpath)
                return None
        else:
            my_print('WARNING: invalid file  ' + str(fpath))
            return None


def write_data_file(data_df, fpath, is_airflow=False):
    # make sure file name has no extensions (it will be appended)
    ext = '.'.join(fpath.split('/')[-1].split('.')[1:])  # supports multiple extensions or none, fname.ext1.ext2.ext3....
    fpath = fpath.replace('.' + ext, '')                 # ext dropped if any

    if is_airflow is True:   # fpath is an s3 file name: s3://BUCKET_NAME/CONTAINER_NAME/FILE_NAME <<< NO EXTENSION!
        return None
        # fname = os.path.basename(fpath)    # drop dirs
        # local_filename = TempUtils.datafile(fname)
        # lpath = write_data_file(data_df, local_filename, is_airflow=False)
        # try:
        #     if os.path.isfile(lpath):
        #         ext = '.'.join(lpath.split('/')[-1].split('.')[1:])
        #         s3fpath = fpath + '.' + ext
        #         s3 = S3Hook()
        #         s3.load_file(lpath, s3fpath, replace=True)
        #         s_ut.my_print('writing file ' + str(lpath) + ' to ' + str(s3fpath))
        #         return s3fpath
        #     else:
        #         s_ut.my_print('could not load ' + str(lpath) + S3')
        #         return None
        # except TypeError:
        #     my_print('could not write to local ' + local_filename)
        #     return None
    else:  # fname is a local file name
        my_print('writing file ' + str(fpath))
        if os.path.dirname(fpath):
            return p_ut.save_df(data_df, fpath, verbose=False)  # returns the path with extension
        else:
            my_print('WARNING: could not write: missing directory ' + str(os.path.dirname(fpath)))
            return None


def my_print(msg, flush=True, fpath=None):
    try:
        caller = getframeinfo(stack()[1][0])
        if fpath is None:
            print("%s:%d - %s" % (caller.filename, caller.lineno, msg), flush=flush)
        else:
            dir = os.path.dirname(fpath)
            if os.path.exists(dir):
                mode = 'a' if os.path.exists(fpath) else 'w'
                string = "%s:%d - %s" % (caller.filename, caller.lineno, msg + '\n')
                with open(fpath, mode) as fp:
                    fp.write(string)
            else:
                print('my_print: invalid path name: ' + str(fpath))
                sys.exit()
    except IndexError:
        print('index error in my_print', flush=True)
        print(msg, flush=True)


def run_cmd(args_list, encoding='utf-8', verbose=True, stdout=None, max_cnt=3):
    _ = do_files('/tmp/', 'daemon', 'remove')
    ret, out, err, p_file = _run_cmd(args_list, encoding, stdout, verbose)
    while ret != 0:
        sys.stdout.write('pid: ' + str(os.getpid()) + ' process failed with error code ' + str(ret) + ' and error message: ' + str(err))
        pid_cnt = do_files('/tmp/', 'daemon', 'count')
        if pid_cnt > max_cnt:
            break
        wait_secs = 600
        print('waiting ' + str(wait_secs) + 'secs before trying again...')
        time.sleep(wait_secs)
        try:
            daemonize(p_file, stdout=None, stderr=None)
        except RuntimeError as e:
            print(e, file=sys.stderr)
            raise SystemExit(1)
        ret, out, err, p_file = _run_cmd(args_list, encoding, stdout, verbose)

    pid_cnt = do_files('/tmp/', 'daemon', 'count')
    _ = do_files('/tmp/', 'daemon', 'remove')
    if pid_cnt > max_cnt:
        print('pid: ' + str(os.getpid()) + ' FAILURE after ' + str(pid_cnt) + ' tries')
        sys.exit(-1)
    else:
        my_print('pid: ' + str(os.getpid()) + ' OK after ' + str(1 + pid_cnt) + ' tries')
        return ret, out, err


def do_files(a_dir, string, op):
    ctr = 0
    for f in os.listdir(a_dir):
        if string in f:
            if op == 'remove':
                os.remove(os.path.join(a_dir, f))
            elif op == 'count':
                ctr += 1
            else:
                return None
    return ctr


def _run_cmd(args_list, encoding, stdout, verbose):
    pidfile = '/tmp/daemon_' + str(os.getpid()) + '.pid'
    if stdout is None:
        stdout = subprocess.PIPE
    if verbose:
        my_print('pid: ' + str(os.getpid()) + ' start cmd: ' + str(args_list) + ' with pid file: ' + str(pidfile))
    completed = subprocess.run(args_list, stdout=stdout, stderr=subprocess.PIPE, encoding=encoding)
    ret, out, err = completed.returncode, completed.stdout, completed.stderr
    return ret, out, err, pidfile


def daemonize(pidfile, *, stdin=None, stdout=None, stderr=None):
    # None in std* to use the parent's
    # Use stdin='/dev/null', stdout='/dev/null', stderr='/dev/null' to go quiet
    ppid = os.getpid()
    print('pid: ' + str(os.getpid()) + ' daemon start from parent ' + str(ppid) + ' file: ' + str(pidfile))
    if os.path.exists(pidfile):
        raise RuntimeError('Already running for ' + str(pidfile))

    # First fork (detaches from parent)
    try:
        if os.fork() > 0:
            raise SystemExit(0)       # Parent exit
    except OSError as e:
        raise RuntimeError('fork #1 failed.')

    os.chdir('/')
    os.umask(0)
    os.setsid()

    # Second fork (relinquish session leadership)
    try:
        if os.fork() > 0:
            raise SystemExit(0)
    except OSError as e:
        raise RuntimeError('fork #2 failed.')

    # Flush I/O buffers
    sys.stdout.flush()
    sys.stderr.flush()

    # Replace file descriptors for stdin, stdout, and stderr
    if stdin is not None:
        with open(stdin, 'rb', 0) as f:
            os.dup2(f.fileno(), sys.stdin.fileno())
    if stdout is not None:
        with open(stdout, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stdout.fileno())
    if stderr is not None:
        with open(stderr, 'ab', 0) as f:
            os.dup2(f.fileno(), sys.stderr.fileno())

    # Write the PID file
    with open(pidfile,'w') as f:
        print(os.getpid(), file=f)

    print('pid: ' + str(os.getpid()) + ' daemon started from parent ' + str(ppid) + ' file: ' + str(pidfile))


# Signal handler for termination (required)
# def sigterm_handler(signo, frame):
#     raise SystemExit(1)
#
#
# signal.signal(signal.SIGTERM, sigterm_handler)


def reap_children(timeout=3, do_sigkill=False):
    """
    Tries hard to terminate and ultimately kill all the children of this process.
    From https://psutil.readthedocs.io/en/latest/#terminate-my-children
    :param timeout:
    :param do_sigkill:
    :return:
    """
    def on_terminate(proc):
        try:
            my_print("process {} terminated with exit code {}".format(proc, proc.returncode))
        except TypeError:
            my_print("process terminated with problems ")

    procs = psutil.Process().children()

    # send SIGTERM
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=timeout, callback=on_terminate)
    if alive:
        for p in alive:
            my_print("process " + str(p) + " survived SIGTERM")

        if do_sigkill is True:    # send SIGKILL
            for p in alive:
                my_print("process " + str(p) + " trying SIGKILL")
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass

            gone, alive = psutil.wait_procs(alive, timeout=timeout, callback=on_terminate)
            if alive:
                # give up
                for p in alive:
                    my_print("process " + str(p) + " survived SIGKILL; giving up")


def do_mp(func, args_arr, is_mp=True, cpus=None, do_sigkill=True, start_method=None, verbose=False, used_cpus=0):
    # start_method: 'spawn', 'fork', 'forkserver'
    if is_mp is True:
        if verbose:
            my_print('pid: ' + str(os.getpid())
                     + ' &&&&&&&&&&&&&&&&&&&&&&&& mp starts:: children: ' + str(len(psutil.Process().children(recursive=True)))
                     + ' start_method: ' + str(mp.get_start_method(allow_none=True)))
        default_method = 'spawn'
        if mp.get_start_method(allow_none=True) is None:
            start_method = default_method if start_method is None else start_method
            mp.set_start_method(start_method, force=True)  # spawn, fork, forkserver
        if cpus is None:
            cpus = min([int(mp.cpu_count()), len(args_arr)])
        else:
            cpus = min([int(cpus), int(mp.cpu_count()), len(args_arr)])
        cpus = max(1, cpus - used_cpus)
        with mp.Pool(processes=int(cpus)) as pool:
            if verbose:
                my_print('pid: ' + str(os.getpid()) + ' pre active children: ' + str(mp.active_children()))
            out_list = pool.starmap(func, args_arr)
        if verbose:
            my_print('pid: ' + str(os.getpid()) + ' post active children: ' + str(mp.active_children()))
        if start_method == 'fork':
            reap_children(do_sigkill=do_sigkill)
            if verbose:
                my_print('pid: ' + str(os.getpid()) + ' fork active children: ' + str(mp.active_children()))
    else:
        out_list = [func(*a) for a in args_arr]
    return [x for x in out_list if x is not None]


