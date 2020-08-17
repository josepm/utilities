__author__ = 'tony_fu'

import os
import sys
import tempfile
import getpass


class TempUtils:
    tmp_root = "{0}/agent_staffing".format(tempfile.gettempdir())
    data_folder = '{0}/data'.format(tmp_root)
    tmp_folder = '{0}/tmp'.format(tmp_root)
    tmp_folder_end_slash = '{0}/'.format(tmp_folder)
    data_folder_end_slash = '{0}/'.format(data_folder)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        os.chmod(tmp_root, 0o777)
        os.chmod(data_folder, 0o777)

    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
        os.chmod(tmp_root, 0o777)
        os.chmod(tmp_folder, 0o777)

    @staticmethod
    def tmpfile(path, is_airflow=True):
        if is_airflow:
            return '{0}/{1}'.format(TempUtils.tmp_folder, path)
        else:
            return os.path.expanduser('~/my_tmp/') + path

    @staticmethod
    def datafile(path, is_airflow=True):
        if is_airflow:
            return '{0}/{1}'.format(TempUtils.data_folder, path)
        else:
            return os.path.expanduser('~/my_data/') + path


