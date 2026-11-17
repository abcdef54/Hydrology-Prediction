"""Stage 3B grid identity, provenance, UTC cycles and synchronized-vector tests."""
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
from research.station_analysis.phase2.spatial_weather.collection import Collector,grid_identity,classify,flatten_records
from research.station_analysis.phase2.spatial_weather.processing import synchronized_features


def payload(cycle,value=1.,lat=16.2,lon=107.2):
    return {'latitude':lat,'longitude':lon,'utc_offset_seconds':0,
            'hourly_units':{'time':'unixtime','precipitation':'mm'},
            'hourly':{'time':[int(t.timestamp()) for t in pd.date_range(cycle+pd.Timedelta(hours=1),periods=48,freq='h')],
                      'precipitation':[value]*48}}


class SpatialWeatherTests(unittest.TestCase):
    def setUp(self):
        self.cycle=pd.Timestamp('2026-01-01T00:00Z')
        self.locations=[{'id':'A','station_id':'A','latitude':16.2,'longitude':107.2},
                        {'id':'B','station_id':'B','latitude':16.21,'longitude':107.21}]

    def test_utc_conversion_before_formatting_run(self):
        with TemporaryDirectory() as temp,patch('time.sleep'):
            collector=Collector(temp,{})
            response=Mock();response.json.return_value=payload(self.cycle)
            collector.client.get=Mock(return_value=response)
            record=collector.fetch(self.locations[:1],self.cycle.tz_convert('Asia/Ho_Chi_Minh'))
            self.assertEqual(record['parameters']['run'],'2026-01-01T00:00')
            self.assertEqual(pd.Timestamp(record['initialized_at']),self.cycle)
            collector.fetch(self.locations[:1],self.cycle)
            self.assertEqual(collector.client.get.call_count,1)

    def test_naive_or_off_cycle_timestamp_never_reaches_network(self):
        with TemporaryDirectory() as temp:
            collector=Collector(temp,{})
            for stamp in ['2026-01-01','2026-01-01T01:00Z']:
                with self.assertRaises(ValueError):collector.fetch(self.locations,stamp)

    def record(self,values):
        return {'initialized_at':str(self.cycle),'retrieved_at':'2026-10-26T00:00Z',
                'locations':self.locations,'parameters':{'cell_selection':'land'},
                'payload':[payload(self.cycle,v) for v in values]}

    def test_same_cell_and_cycle_deduplicated_with_mapping_preserved(self):
        runs,points,mapping=flatten_records([self.record([1.,1.])])
        self.assertEqual(len(runs),1);self.assertEqual(len(points),48);self.assertEqual(len(mapping),2)
        self.assertEqual(mapping.grid_id.nunique(),1)
        self.assertEqual(runs.iloc[0].provenance_class,'hindcast/reconstructed')
        self.assertTrue(pd.isna(runs.iloc[0].available_at))

    def test_conflicting_same_cell_run_values_are_not_silently_averaged(self):
        with self.assertRaises(ValueError):flatten_records([self.record([1.,2.])])

    def test_reordered_provider_locations_cannot_silently_remap_stations(self):
        record=self.record([1.,1.])
        record['payload'][0]['location_id']=1
        with self.assertRaises(ValueError):flatten_records([record])

    def test_identity_changes_when_returned_grid_changes(self):
        self.assertNotEqual(grid_identity('ecmwf_ifs',16.2,107.2),grid_identity('ecmwf_ifs',16.3,107.2))
        self.assertNotEqual(grid_identity('ecmwf_ifs',16.2,107.2),grid_identity('other',16.2,107.2))

    def test_version_transition_does_not_claim_observed_availability(self):
        self.assertEqual(classify('2026-05-12T00:00Z')[1],'hindcast/reconstructed')
        self.assertEqual(classify('2026-05-12T06:00Z')[1],'assumed_availability')

    def test_common_initialization_falls_back_for_all_grids_together(self):
        anchor=self.cycle+pd.Timedelta(hours=12)
        runs=[];points=[]
        for gid in ['A','B']:
            for age in [12,6]:
                cycle=anchor-pd.Timedelta(hours=age);rid=f'{gid}{age}'
                runs.append({'location_id':gid,'initialized_at':cycle,'retrieved_at':cycle+pd.Timedelta(days=300),
                             'run_id':rid,'available_at':pd.NaT,'archive_kind':'hindcast_49r1'})
                for h in range(1,49):
                    if gid=='B' and age==6 and h==9:continue
                    points.append({'run_id':rid,'valid_at':cycle+pd.Timedelta(hours=h),'precipitation_mm':float(age),'quality_status':'forecast'})
        result=synchronized_features(pd.DataFrame(),['A','B'],pd.DatetimeIndex([anchor]),
                                     run_audit=pd.DataFrame(runs),points=pd.DataFrame(points))
        self.assertEqual(result.iloc[0].initialized_at,self.cycle)
        self.assertEqual(result.iloc[0]['A|rain_0_3h_mm'],36.)
        self.assertEqual(result.iloc[0]['B|rain_0_3h_mm'],36.)

class TrainOnlyGridRecommendationTests(unittest.TestCase):
    def test_evaluation_scores_do_not_change_candidate_arms(self):
        from research.station_analysis.phase2.spatial_weather.analysis import recommendations
        grids=pd.DataFrame({'grid_id':['A','B','C'],'grid_latitude':[16.274164,16.48,16.2],
                            'grid_longitude':[107.26496,107.57,107.85]})
        stations=pd.DataFrame([
            {'station_id':'W','name':'Water','station_type':'water_level','latitude':16.45,'longitude':107.55},
            {'station_id':'R1','name':'Rain 1','station_type':'rainfall','latitude':16.274,'longitude':107.265},
            {'station_id':'R2','name':'Rain 2','station_type':'rainfall','latitude':16.275,'longitude':107.265},
        ])
        candidates=pd.DataFrame({'water_station_id':['W','W'],'rain_station_id':['R1','R2'],'best_corr_dh12':[.4,.3]})
        partial=pd.DataFrame({'water_station_id':['W']*4,'scope':['train','train','evaluation','evaluation'],
                              'control_set':['plus_existing_a_luoi']*4,'grid_id':['B','C','B','C'],
                              'partial_pearson_r':[.2,.1,.01,.9]})
        before=recommendations(grids,stations,candidates,partial,pd.DataFrame(),'2026-08-01T00:00+07:00')
        partial.loc[partial.scope.eq('evaluation'),'partial_pearson_r']=[1000.,-1000.]
        after=recommendations(grids,stations,candidates,partial,pd.DataFrame(),'2026-08-01T00:00+07:00')
        pd.testing.assert_frame_equal(before,after)
        self.assertEqual(before.loc[before.arm.eq('W3'),'grid_ids'].iloc[0],'A;B')


if __name__=='__main__':unittest.main()
