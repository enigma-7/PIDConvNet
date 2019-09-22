//
// created:       2016-10-07
// last changed:  2016-10-07
//
// based on AliTRDcheckConfig.C
//


#if !defined(__CINT__) || defined(__MAKECINT__)

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <TFile.h>
#include <TString.h>

#include <AliTRDCalDet.h>
#include <AliTRDSensorArray.h>
#include <TCanvas.h>
#include <AliGRPObject.h>
#include <AliCDBEntry.h>
#include <AliTRDCalDCS.h>
#include <AliTRDCalDCSv2.h>
#include <AliCDBManager.h>

#endif

using std::cout;
using std::endl;

struct runinfo_t
{

  // GRP info
  TTimeStamp start_time;
  TTimeStamp end_time;

  float  cavern_pressure;
  //float  surface_pressure;
  //int    detector_mask;

  float anode_voltage[540];
  float drift_voltage[540];

  float local_gain_factor[540][16][144];
  float online_local_gain_factor[540][16][144];

  float gain[540];
  float vdrift[540];
  float ExB[540];
  float status[540];
  float T0[540];

};

//__________________________________________________________________________
void pad_query(Int_t year=2016, Int_t run=265377)
{

    ofstream chamber_info_file;
    chamber_info_file.open(Form("calib_files/chamber_info_%d_%d.txt", year, run));

    ofstream local_gains_file;
    local_gains_file.open(Form("calib_files/local_gains_%d_%d.txt", year, run));

  // set up the connection to the OCDB
  AliCDBManager* man = AliCDBManager::Instance();

  AliTRDcalibDB* const calibration = AliTRDcalibDB::Instance();

  if (0) {
    man->SetDefaultStorage
      (Form("local:///cvmfs/alice-ocdb.cern.ch/calibration/data/%d/OCDB/",year));
  } else {
    man->SetDefaultStorage(Form("alien://folder=/alice/data/%d/OCDB/",year));
  }

  man->SetCacheFlag(kTRUE);
  man->SetRun(run);

  calibration->SetRun(run);

  runinfo_t runinfo;

  // -----------------------------------------------------------------------
  // Get the GRP data. Only runs with a corresponding GRP entry in the OCDB
  // will be processed
  AliCDBEntry *entry = man->Get("GRP/GRP/Data",run);
  //AliGRPObject* grp = (AliGRPObject*)entry->GetObject();
  AliGRPObject* grp = (AliGRPObject*)entry->GetObject();

  if (!grp) return;

  TTimeStamp start (grp->GetTimeStart());
  TTimeStamp end   (grp->GetTimeEnd());

  runinfo.start_time = start;
  runinfo.end_time = end;
  runinfo.cavern_pressure =
    grp->GetCavernAtmosPressure()->MakeGraph()->GetMean();
  //runinfo.surface_pressure = grp->GetSurfaceAtmosPressure();
  //runinfo.detector_mask = grp->GetDetectorMask();

  if (0) {
    TCanvas* c = new TCanvas("cavern pressure", "cavern pressure");
    grp->GetCavernAtmosPressure()->MakeGraph()->Draw("alp");
  }

  // -----------------------------------------------------------------------
  // HV data

  entry = man->Get("TRD/Calib/trd_hvAnodeUmon",run);
  AliTRDSensorArray* arr = (AliTRDSensorArray*)entry->GetObject();

  bool tmp = true;

  for (int i=0;i<arr->NumSensors();i++) {

    //cout << arr->GetSensorNum(i)->GetIdDCS() << "  "
    //<< arr->GetSensorNum(i)->GetStringID() << endl;

    //cout << arr->GetSensorNum(i)->Eval(start,true) << endl;;

    // somehow, this array does not produce a TGraph, therefore we use
    // the value at the start of the run
    runinfo.anode_voltage[i] = arr->GetSensorNum(i)->Eval(start,tmp);
  }

  entry = man->Get("TRD/Calib/trd_hvDriftUmon",run);
  arr = (AliTRDSensorArray*) entry->GetObject();

  for (int i=0;i<arr->NumSensors();i++) {
    // somehow, this array does not produce a TGraph, therefore we use
    // the value at the start of the run
    runinfo.drift_voltage[i] = arr->GetSensorNum(i)->Eval(start,tmp);
  }

  // get calibration information

  entry = man->Get("TRD/Calib/ChamberGainFactor",run);
  AliTRDCalDet* g = (AliTRDCalDet*)entry->GetObject();

  for (int i=0; i<540; i++) {
    runinfo.gain[i] = g->GetValue(i);
  }

  entry = man->Get("TRD/Calib/ChamberVdrift",run);
  AliTRDCalDet* vd = (AliTRDCalDet*)entry->GetObject();

  for (int i=0; i<540; i++) {
    runinfo.vdrift[i] = vd->GetValue(i);
  }

  entry = man->Get("TRD/Calib/ChamberExB",run);
  AliTRDCalDet* exb = (AliTRDCalDet*)entry->GetObject();

  for (int i=0; i<540; i++) {
    runinfo.ExB[i] = exb->GetValue(i);
  }

  entry = man->Get("TRD/Calib/ChamberStatus",run);
  AliTRDCalDet* stat = (AliTRDCalDet*)entry->GetObject();
  for (int i=0; i<540; i++) {
    runinfo.status[i] = stat->GetValue(i);
  }

  entry = man->Get("TRD/Calib/ChamberT0",run);
  AliTRDCalDet* t0 = (AliTRDCalDet*)entry->GetObject();
  for (int i=0; i<540; i++) {
    runinfo.T0[i] = t0->GetValue(i);
  }

  entry = man->Get("TRD/Calib/LocalGainFactor", run);
  AliTRDCalPad* loc = (AliTRDCalPad*)entry->GetObject();

  for (int d=0; d<540; d++) {
    AliTRDCalROC *det_cal_roc = loc->GetCalROC(d);
    fCalOnlGainROC = calibration->GetOnlineGainTableROC(d);
    for (int r=0; r<16; r++) {
      for (int c=0; c<144; c++) {
        if (r >= det_cal_roc->GetNrows()) {
            runinfo.local_gain_factor[d][r][c] = 1.0;
            runinfo.online_local_gain_factor[d][r][c] = 1.0;
            continue;
        }
        runinfo.local_gain_factor[d][r][c] = det_cal_roc->GetValue(c,r);

        if(fCalOnlGainROC)
            runinfo.online_local_gain_factor[d][r][c] = fCalOnlGainROC->GetGainCorrectionFactor(r, c);
        else
            runinfo.online_local_gain_factor[d][r][c] = 0;
      }
    }
  }

  cout << "      Start time : " << runinfo.start_time << endl;
  cout << "        End time : " << runinfo.end_time << endl;
  cout << " Cavern pressure : " << runinfo.cavern_pressure << endl;


  for (int d=0; d<540; d++) {

    if (d < 5) {
        cout << "chamber " << d << endl;
        cout << "      anode voltage : " << runinfo.anode_voltage[d] << endl;
        cout << "      drift voltage : " << runinfo.drift_voltage[d] << endl;
        cout << "               gain : " << runinfo.gain[d] << endl;
        cout << "     drift velocity : " << runinfo.vdrift[d] << endl;
        cout << "                ExB : " << runinfo.ExB[d] << endl;
        cout << "  local_gain_factor : " << runinfo.local_gain_factor << endl;
        cout << "  local_gain_factor : " << runinfo.local_gain_factor[d] << endl;
        cout << "  local_gain_factor[0][0] : " << runinfo.local_gain_factor[d][0][0] << endl;
        cout << "  online_local_gain_factor[0][0] : " << runinfo.online_local_gain_factor[d][0][0] << endl;
    }

    chamber_info_file << d << ", " << runinfo.anode_voltage[d] << ", " << runinfo.drift_voltage[d] << ", " << runinfo.gain[d] << ", " << runinfo.vdrift[d] << ", " << runinfo.ExB[d] << std::endl;

    for (int r=0; r<16; r++) {
      local_gains_file << d << ", " << r;

      for (int c=0; c<144; c++) {
         local_gains_file << ", " << runinfo.local_gain_factor[d][r][c];
      }
      local_gains_file << std::endl;
    }
  }

  chamber_info_file.close();
  local_gains_file.close();
}
