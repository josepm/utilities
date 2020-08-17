__author__ = 'tony_fu'

import os
from retrying import retry
from datetime import datetime
import uuid
import socket
import sys
import platform

from capacity_planning.data import do_hql as hql
from capacity_planning.utilities.temp_utils import TempUtils


class TaskTracker:
    def __init__(self, argv):
        if len(argv) > 2:
            self.run_id = argv[1]
        else:
            self.run_id = 'fake_id'
        self.total_shards = -1
        self.step = argv[0].split('/')[-1].split('.')[0]
        self.shard_id = socket.gethostname()
        self.command = ' '.join(argv)

    def set_status(self, status):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        query = r"""DELETE FROM josep.airflow_task_status_tracking
            WHERE run_id = '{self.run_id}' and step = '{self.step}' and shard_id = '{self.shard_id}';
            INSERT INTO josep.airflow_task_status_tracking VALUES(
                '{self.command}',
                '{status}',
                {self.total_shards},
                '{timestamp}',
                '{self.run_id}',
                '{self.step}',
                '{self.shard_id}'
            );""".format(**locals())

        fquery = TempUtils.tmpfile(str(self.shard_id) + '_' + str(os.getpid()) + '_' + 'tracking.hql')

        print('++ airflow_task_tracking::saving query ' + query + ' to file ' + fquery)
        try:
            with open(fquery, 'w') as fp:
                fp.write(query)
            is_ap = True if platform.system() == 'Darwin' else False
            fout = hql.exec_query(fquery, 'presto', expected_data=False)
        except PermissionError:
            print('ERROR: TaskTracker could not open ' + fquery)

        if status == 'success':
            print("SUCCESS")
        elif status == 'failure':
            print('FAILURE')
        else:
            pass

    def my_exit(self, ret):
        if ret == 0:
            if platform.system() != 'Darwin':
                self.set_success()
        else:
            if platform.system() != 'Darwin':
                self.set_failure()
            raise RuntimeError('failure')
        sys.exit(0)

    def set_started(self):
        self.set_status('started')

    def set_success(self):
        self.set_status('success')

    def set_failure(self):
        self.set_status('failure')

    @staticmethod
    def gen_run_id():
        return str(uuid.uuid4())

    @staticmethod
    @retry(stop_max_attempt_number=3)
    def wait_all_shards_to_complete(run_id, hosts, step):
        # import airpy as ap
        # query = r"""
        #     SELECT shard_id, status
        #     FROM josep.airflow_task_status_tracking
        #     WHERE run_id = '{run_id}' AND step = '{step}'
        # """.format(**locals())
        #
        # ap.config.ldap_username = 'tony_fu'
        # ap.config.save()
        #
        # df = ap.presto(query, renew=True)
        # print("++ run_all:Query shard result {query}".format(**locals()))
        # print(df)
        #
        # success = 0
        # failure = 0
        # for index, row in df.iterrows():
        #     if row['status'] == "success":
        #         success += 1
        #     elif row['status'] == "failure":
        #         failure += 1
        #
        # return success, failure

        return 1