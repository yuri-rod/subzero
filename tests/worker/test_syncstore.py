import pytest

from subzero.worker.jobs import JobStore
from subzero.worker.syncstore import SyncStore


def test_failed_download_reservations_survive_restart(tmp_path):
    jobs=JobStore(str(tmp_path/'jobs.db'))
    state=SyncStore(jobs)
    assert state.reserve('video','pt-BR',17,'job')
    state.update('video','pt-BR',17,status='rejected')
    restarted=SyncStore(jobs)
    assert not restarted.reserve('video','pt-BR',17,'job2')
    assert restarted.reserve('other','pt-BR',18,'job2')
    assert len(restarted.attempts('video','pt-BR')) == 1


def test_audit_invalidates_on_subtitle_digest_change(tmp_path):
    state=SyncStore(JobStore(str(tmp_path/'jobs.db')))
    state.audit('video','pt-BR','hash1','pass',{'reason':'ok'})
    assert state.current('video','pt-BR','hash1')
    assert not state.current('video','pt-BR','hash2')


def test_sql_values_are_not_interpolated(tmp_path):
    state=SyncStore(JobStore(str(tmp_path/'jobs.db')))
    key="x'; DROP TABLE jobs;--"
    state.audit(key,'pt-BR','a','reject',{})
    assert state.current(key,'pt-BR','a')


def test_broken_file_ids_lists_only_broken_status(tmp_path):
    from subzero.worker.syncstore import broken_file_ids
    jobs=JobStore(str(tmp_path/'jobs.db'))
    state=SyncStore(jobs)
    state.reserve('v','pt-BR',1,'job')
    state.update('v','pt-BR',1,status='broken')
    state.reserve('v','pt-BR',2,'job')
    state.update('v','pt-BR',2,status='reject')
    state.reserve('w','pt-BR',3,'job')

    assert broken_file_ids(jobs,'pt-BR') == {1}
    assert broken_file_ids(jobs,'en') == set()
    assert broken_file_ids(None,'pt-BR') == set()
