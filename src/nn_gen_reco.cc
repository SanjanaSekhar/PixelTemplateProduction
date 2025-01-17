/*
 * gen_zp_template2d.cc
 * Original by Morris Swartz
 * Updated by Oz Amram          May 2019
 *
 * Create template from previous generated files.
 * Make error objects by calling 2d reco with created templates.

 *   Change notes:
 *   Template hit reconstruction algorithms, add FPix to templates, add double pixels to templates, 
 *   change dp calling seq (Add PSI46v2 response function and allow for zero-suppressed ROC output)
 *   Change to Root historgraams for long-term compatibility
 *   Add angle vs resolution for templates and "Standard Algorithms"
 *   Tune CMSSW simulation for template 1 reconstruction
 *   Change standard algorithm to always use edge method for y-reconstruction
 *   Add Estar template number 4
 *   Do cosmics
 *   Add clustering  algorithm
 *   Change response function, parameters to that used for real analysis
 *   Same as test_code_v9 but with weighting for flat eta distributions in the resolution histograms
 *   Add 2-threshold simulation
 *   Work on templates with non-zero scale factors, ask for template IDs to use
 *   Low seed threshold verion for thin pixels
 *   Remove generic reco to avoid needing generror objects
 *   Use CMSSW data structures from VI
 *   Write integers for the templates
 */

// Modifying to include NN_reco

#define TEMPL_DEBUG
#include "template_utils.h"
//#include "nn_reco.h"

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"
#include "tensorflow/core/platform/env.h"
#include <chrono>
using namespace std::chrono;

using namespace tensorflow;

// Main program  

int main(int argc, char *argv[])
{

    // Local variables 
    std::vector<float> pvec(6), wgauss(TYSIZE), vgauss(TYSIZE), xgauss(TYSIZE), ygauss(TYSIZE), zgauss(TYSIZE);
    float pixin[TXSIZE][TYSIZE];
    float ytemp[9][TYSIZE], xtemp[9][TXSIZE], xpar[2][5], ypar[2][5];
    float sxmax, symax, sxmaxx, symaxx, cosx, cosy, cosz;
    static float thick, xsize, ysize, noise, zcen, gain_frac, q100_frac, common_frac, readout_noise, qscale;
    static float qavg_raw,  clslnx, clslny; 
    static float xrec, yrec, sigmax, sigmay, probx, proby, probQ,  signal, locBz, locBx,  pixmax;
    static float pixmaxy, pixmaxx;
    static int startfile,  nbad, ngood, fe_model_type, numrun; 
    int  id,NTy, NTyx,NTxx,IDtype;


    std::multiset<float> qmsort;
    static float fbin[] = {1.5, 1.0, 0.85}; //bins of charge in Q_cluster / Q_avg

    const int nevents = 30;

    float xhit[nevents], yhit[nevents], cotalpha, cotbeta;

    float  qsmear[nevents], npix[nevents],  qflx[nevents], qfly[nevents],
    qtotal[nevents];
    
    bool good_clust[nevents];

    int nelec[nevents], qbins[nevents], qbin_merge[nevents], xwidth[nevents], xstart[nevents], ywidth[nevents], ystart[nevents];

    float ***cluster = setup_3d_array(nevents, TXSIZE, TYSIZE); //Cluster as found by seeding + clustering algo
    float **xsum1 = setup_2d_array(nevents, TXSIZE);
    float **xsum2 = setup_2d_array(nevents, TXSIZE);

    float **ysum1 = setup_2d_array(nevents, TYSIZE);
    float **ysum2 = setup_2d_array(nevents, TYSIZE);
    static float Bfield,Vbias,temp,fluenc;
    float nqbin[5];
    double dx, dy;
    static float q100, q101, q50, q51,  qmax; 
    float xtalk_frac, xtalk_noise;


    //parameters for template fit
    SiPixelTemplateEntry * slice;
    bool ydouble[TYSIZE], xdouble[TXSIZE];
    memset(ydouble, false, sizeof(ydouble));
    memset(xdouble, false, sizeof(xdouble));

    int mrow = TXSIZE, mcol = TYSIZE;


    const float fmax = 0.5f;
    int write_temp_header, use_l1_offset;

    const double rten = 10.;

    const int nvers = 21;

    float qin;
    static char infile[120], label[160], header[120], outfile0[120], outfile1[120], outfile2[120];
    //	int random(void);

    float clust[TXSIZE][TYSIZE], rclust[TXSIZE][TYSIZE], sigraw[TXSIZE+2][TYSIZE+2];
    bool bclust[TXSIZE][TYSIZE];
    std::pair<int, int> pixel, max;

    FILE *temp_output_file, *generr_output_file;

    struct timeval now0, now1, t_1dcnn1,t_1dcnn2;
    struct timezone timz;
    long deltas, deltaus;
    double deltat;

    TCanvas* c1 = new TCanvas("c1", header, 1600, 1000);
    c1->SetFillStyle(4000);


    //  Read which data and inputs to use (use c file i/o which is much faster than c++ i/o) 

    FILE *config_file = fopen("pix_2t.proc", "r");
    if (config_file==NULL) {
        printf("no pixel initialization file found \n");
        return 0;
    }


    char extra[80];
    char line[160];
    fgets(line, 160, config_file);

    int num_read = sscanf(line,"%d %d %f %f %f %f %f %f %f %d %s", &startfile, &numrun, &noise, &q100, 
            &q101, &q100_frac, &common_frac, &gain_frac, &readout_noise, &fe_model_type, &extra[0]);
    printf("processing %d files starting from %d, noise = %f, threshold0 = %f, threshold1 = %f," 
            "rms threshold frac = %f, common_frac = %f, gain fraction = %f, readout noise = %f, front end model type = %d, extra = %s \n", 
            numrun, startfile, noise, q100, q101, q100_frac, common_frac, gain_frac, readout_noise, fe_model_type, extra);
    if(num_read < 10){
        printf("Error reading config file !. Only read %i params \n", num_read);
        return 0;
    }


    fgets(line, 160, config_file);
    num_read = sscanf(line, " %d %d %f %f", &use_l1_offset, &write_temp_header, &xtalk_frac, &xtalk_noise);
    if(num_read != 4){
        printf("Error reading config file !\n");
        printf("Line was %s \n", line);
        return 0;
    }
    fgets(line, 160, config_file);
    num_read = sscanf(line, " %d %d %d %d %d %f %f %f %f %f",  &id, &NTy, &NTyx,&NTxx, &IDtype, &Bfield, &Vbias, &temp, &fluenc, &qscale);
    printf("Using params: Use_l1_offset=%d, write_temp_header=%d, ID=%d NTy=%d NTyx=%d NTxx=%d Dtype=%d Bfield=%.2f "
            "Bias Voltage = %.1f temparature = %.0f fluence = %.2f q-scale = %.4f xtalk_frac=%.2f  xtalk_noise = %.2f \n",
            use_l1_offset, write_temp_header, id, NTy, NTyx, NTxx, IDtype, Bfield, Vbias, temp, fluenc, qscale, xtalk_frac, xtalk_noise);
    if(num_read != 10){
        printf("Error reading config file !\n");
        printf("Only %i params when there should be 10 \n",num_read); 
        printf("Line was %s \n", line);
        return 0;
    }

    fclose(config_file);

    FrontEndModel frontEnd;
    frontEnd.fe_type       = fe_model_type;
    frontEnd.threshold = q100;
    frontEnd.gain_frac     = gain_frac;
    frontEnd.readout_noise = readout_noise;
    if(use_l1_offset) {
        printf("using L1 parameters \n");
        frontEnd.vcal = 50.;	
        frontEnd.vcaloffst = 670.;
    }


    //  Calculate 50% of threshold in q units and enc noise in adc units

    q50=0.5*q100;
    q51=0.5*q101;

    //  Open template output file

    sprintf(infile,"template_summary_zp%5.5d.out",id);
    temp_output_file = fopen(infile, "w");
    if (temp_output_file==NULL) {
        printf("couldn't open template output file/n");
        return 0;
    }

    sprintf(infile,"generror_summary_nnzp%5.5d.out",id);
    generr_output_file = fopen(infile, "w");
    if (generr_output_file==NULL) {
        printf("couldn't open generr output file/n");
        return 0;
    }

    //  Open Lorentz summary file and read stored quantities

    sprintf(infile,"lorentz_widths_new%5.5d.out",startfile);
    FILE *lorw_file = fopen(infile, "r");
    if (lorw_file==NULL) {
        printf("couldn't find Lorentz summary file/n");
        return 0;
    }
    float lorwdy, lorbsy, lorwdx, lorbsx;
    fscanf(lorw_file,"%f %f %f %f", &lorwdy, &lorbsy, &lorwdx, &lorbsx);
    fclose(lorw_file);
    //  Flip the signs to convert from pixelav to cmssw conventions
    lorwdy = -lorwdy;
    lorbsy = -lorbsy;
    lorwdx = -lorwdx;
    lorbsx = -lorbsx;


    // Define the histograms to be used at each angle pair

    double halfxs=300.;
    int nx=120;
    double halfys=300.;
    int ny=120;
    double chimx=48.;


    const int y_2dcnn_idx =0;
    const int y_dnn_idx =5;
    const int x_2dcnn_idx =10;
    const int x_dnn_idx =15;

    const int y_temp_fp_idx =20;
    const int x_temp_fp_idx =25;
    const int y_generic_idx =30;
    const int x_generic_idx =40;
    const int charge_idx = 50;
    const int y_chi2_fp_idx = 58;
    const int x_chi2_fp_idx = 62;
    const int y_1dcnn_idx = 70;
    const int x_1dcnn_idx = 75;

    const int y_corr_idx = 0;
    const int x_corr_idx = 5;

    const int n_hists = 90; // 70->90
    const int n_profs = 10;


    gStyle->SetOptFit(101);
    gStyle->SetHistLineWidth(2);
    static vector<TH1F*> hp(n_hists);
    static vector<TProfile*> profs(n_profs);
    double chi_min[n_hists]; //chi2 minimum values, keep same indexing


    

    hp[y_generic_idx + 0] = new TH1F("h406","dy_generic (all sig); #Deltay (#mum)",ny,-halfys,halfys);
    hp[y_generic_idx + 1] = new TH1F("h407","dy_generic (signal > 1.5mn); #Deltay (#mum)",ny,-halfys,halfys);      
    hp[y_generic_idx + 2] = new TH1F("h408","dy_generic (1.5mn > signal > 1.0mn); #Deltay (#mum)",ny,-halfys,halfys);      
    hp[y_generic_idx + 3] = new TH1F("h409","dy_generic (1.0mn > signal > 0.85mn); #Deltay (#mum)",ny,-halfys,halfys);     
    hp[y_generic_idx + 4] = new TH1F("h410","dy_generic (0.85mn > signal); #Deltay (#mum)",ny,-halfys,halfys);      

    hp[x_generic_idx + 0] = new TH1F("h401","dx_generic (all sig); #Deltax (#mum)",nx,-halfxs,halfxs);
    hp[x_generic_idx + 1] = new TH1F("h402","dx_generic (signal > 1.5mn); #Deltax (#mum)",nx,-halfxs,halfxs);      
    hp[x_generic_idx + 2] = new TH1F("h403","dx_generic (1.5mn > signal > 1.0mn); #Deltax (#mum)",nx,-halfxs,halfxs);      
    hp[x_generic_idx + 3] = new TH1F("h404","dx_generic (1.0mn > signal > 0.85mn); #Deltax (#mum)",nx,-halfxs,halfxs);     
    hp[x_generic_idx + 4] = new TH1F("h405","dx_generic (0.85mn > signal); #Deltax (#mum)",nx,-halfxs,halfxs);    


    hp[charge_idx + 0] = new TH1F("h100","Number generated e",150,0.,500000.);	
    hp[charge_idx + 1] = new TH1F ("h500","Cluster Charge",250,0.,500000.);
    hp[charge_idx + 2] = new TH1F ("h501","npix(signal > 1.5mn)",40,0.5,40.5);
    hp[charge_idx + 3] = new TH1F ("h502","npix(1.5mn > signal > 1.0mn)",40,0.5,40.5);
    hp[charge_idx + 4] = new TH1F ("h503","npix(1.0mn > signal > 0.85mn)",40,0.5,40.5);
    hp[charge_idx + 5] = new TH1F ("h504","npix(0.85mn > signal)",40,0.5,40.5);
    hp[charge_idx + 6] = new TH1F ("h505","2 Cluster Merged Charge",500,0.,1000000.);
    hp[charge_idx + 7] = new TH1F ("h606","measured Q/generated Q",300,0.,1.5);

 //======= NNs =========   
    hp[y_1dcnn_idx + 0] = new TH1F("h706","dy_1dcnn (all sig); #Deltay (#mum)",ny,-halfys,halfys);
    hp[y_1dcnn_idx + 1] = new TH1F("h707","dy_1dcnn (signal > 1.5mn); #Deltay (#mum)",ny,-halfys,halfys);      
    hp[y_1dcnn_idx + 2] = new TH1F("h708","dy_1dcnn (1.5mn > signal > 1.0mn); #Deltay (#mum)",ny,-halfys,halfys);      
    hp[y_1dcnn_idx + 3] = new TH1F("h709","dy_1dcnn (1.0mn > signal > 0.85mn); #Deltay (#mum)",ny,-halfys,halfys);     
    hp[y_1dcnn_idx + 4] = new TH1F("h710","dy_1dcnn (0.85mn > signal); #Deltay (#mum)",ny,-halfys,halfys);      

    hp[x_1dcnn_idx + 0] = new TH1F("h701","dx_1dcnn (all sig); #Deltax (#mum)",nx,-halfxs,halfxs);
    hp[x_1dcnn_idx + 1] = new TH1F("h702","dx_1dcnn (signal > 1.5mn); #Deltax (#mum)",nx,-halfxs,halfxs);      
    hp[x_1dcnn_idx + 2] = new TH1F("h703","dx_1dcnn (1.5mn > signal > 1.0mn); #Deltax (#mum)",nx,-halfxs,halfxs);      
    hp[x_1dcnn_idx + 3] = new TH1F("h704","dx_1dcnn (1.0mn > signal > 0.85mn); #Deltax (#mum)",nx,-halfxs,halfxs);     
    hp[x_1dcnn_idx + 4] = new TH1F("h705","dx_1dcnn (0.85mn > signal); #Deltax (#mum)",nx,-halfxs,halfxs);    

/*
    hp[y_2dcnn_idx + 0] = new TH1F("h106","dy_2dcnn (all sig); #Deltay (#mum)",ny,-halfys,halfys);
    hp[y_2dcnn_idx + 1] = new TH1F("h107","dy_2dcnn (signal > 1.5mn); #Deltay (#mum)",ny,-halfys,halfys);      
    hp[y_2dcnn_idx + 2] = new TH1F("h108","dy_2dcnn (1.5mn > signal > 1.0mn); #Deltay (#mum)",ny,-halfys,halfys);      
    hp[y_2dcnn_idx + 3] = new TH1F("h109","dy_2dcnn (1.0mn > signal > 0.85mn); #Deltay (#mum)",ny,-halfys,halfys);     
    hp[y_2dcnn_idx + 4] = new TH1F("h110","dy_2dcnn (0.85mn > signal); #Deltay (#mum)",ny,-halfys,halfys);      

    hp[x_2dcnn_idx + 0] = new TH1F("h101","dx_2dcnn (all sig); #Deltax (#mum)",nx,-halfxs,halfxs);
    hp[x_2dcnn_idx + 1] = new TH1F("h102","dx_2dcnn (signal > 1.5mn); #Deltax (#mum)",nx,-halfxs,halfxs);      
    hp[x_2dcnn_idx + 2] = new TH1F("h103","dx_2dcnn (1.5mn > signal > 1.0mn); #Deltax (#mum)",nx,-halfxs,halfxs);      
    hp[x_2dcnn_idx + 3] = new TH1F("h104","dx_2dcnn (1.0mn > signal > 0.85mn); #Deltax (#mum)",nx,-halfxs,halfxs);     
    hp[x_2dcnn_idx + 4] = new TH1F("h105","dx_2dcnn (0.85mn > signal); #Deltax (#mum)",nx,-halfxs,halfxs);      

    hp[y_dnn_idx + 0] = new TH1F("h206","dy_dnn (all sig); #Deltay (#mum)",ny,-halfys,halfys);
    hp[y_dnn_idx + 1] = new TH1F("h207","dy_dnn (signal > 1.5mn); #Deltay (#mum)",ny,-halfys,halfys);
    hp[y_dnn_idx + 2] = new TH1F("h208","dy_dnn (1.5mn > signal > 1.0mn); #Deltay (#mum)",ny,-halfys,halfys);
    hp[y_dnn_idx + 3] = new TH1F("h209","dy_dnn (1.0mn > signal > 0.85mn); #Deltay (#mum)",ny,-halfys,halfys);
    hp[y_dnn_idx + 4]=  new TH1F("h210","dy_dnn (0.85mn > signal); #Deltay (#mum)",ny,-halfys,halfys);

    hp[x_dnn_idx + 0] = new TH1F("h201","dx_dnn (all sig); #Deltax (#mum)",nx,-halfxs,halfxs);
    hp[x_dnn_idx + 1] = new TH1F("h202","dx_dnn (signal > 1.5mn); #Deltax (#mum)",nx,-halfxs,halfxs);
    hp[x_dnn_idx + 2] = new TH1F("h203","dx_dnn (1.5mn > signal > 1.0mn); #Deltax (#mum)",nx,-halfxs,halfxs);
    hp[x_dnn_idx + 3] = new TH1F("h204","dx_dnn (1.0mn > signal > 0.85mn); #Deltax (#mum)",nx,-halfxs,halfxs);
    hp[x_dnn_idx + 4]=  new TH1F("h205","dx_dnn (0.85mn > signal); #Deltax (#mum)",nx,-halfxs,halfxs);
  */
//========================

  



    // Set style for the the histograms	

    for(unsigned int i=0; i<hp.size(); ++i) {
        if(hp[i] == NULL) continue;
        hp[i]->SetLineColor(2);
        hp[i]->SetFillColor(38);
    }



    std::vector<std::pair<int, int> > pixlst;

    // Create template object

    std::vector< SiPixelTemplateStore > thePixelTemp_;
    SiPixelTemplate templ(thePixelTemp_);

    //  Set the ID to -1 to flag the special reco mode

    int tempID = -1;

    //  Determine current time

    gettimeofday(&now0, &timz);


    //==========================================================================================
    // FOR NN RECO
    char *graph_ext = "1dcnn_p1_apr12";
   char graph_x[100],graph_y[100], inputTensorName_x[100], anglesTensorName_x[100],inputTensorName_y[100], anglesTensorName_y[100],outputTensorName_[100];
   sprintf(graph_x,"data/graph_x_%s.pb",graph_ext);
   sprintf(graph_y,"data/graph_y_%s.pb",graph_ext) ;

   //printf("TXSIZE = %i\n", TXSIZE);
   //printf("TYSIZE = %i\n", TYSIZE);
   
   sprintf(inputTensorName_x,"input_1");
   sprintf(anglesTensorName_x,"input_2");
   sprintf(inputTensorName_y,"input_3");
   sprintf(anglesTensorName_y,"input_4");
   sprintf(outputTensorName_,"Identity"); 

   GraphDef graphDef_x;
  Session* session_x;
  Status status; SessionOptions sessionOptions;
 status = NewSession(sessionOptions, &session_x);

  GraphDef graphDef_y;
  Session* session_y;
   status = NewSession(sessionOptions, &session_y);

  std::vector<tensorflow::Tensor> output_x;
  std::vector<tensorflow::Tensor> output_y;

 
  //=========== infer x ====================
  // load the graph
   status = ReadBinaryProto(Env::Default(), graph_x, &graphDef_x);
   // create a new session and add the graphDef
  status = session_x->Create(graphDef_x);
   // define a tensor and fill it with cluster projection
  tensorflow::Tensor cluster_flat_x(tensorflow::DT_FLOAT, {1,TXSIZE,1});

 //=========== infer y ====================
  // load the graph
   status = ReadBinaryProto(Env::Default(), graph_y, &graphDef_y);
   // create a new session and add the graphDef
  status = session_y->Create(graphDef_y);
   // define a tensor and fill it with cluster projection
  tensorflow::Tensor cluster_flat_y(tensorflow::DT_FLOAT, {1,TYSIZE,1});

  // angles
  tensorflow::Tensor angles(tensorflow::DT_FLOAT, {1,2});

    //========================================================================================

    // Loop over angle pair runs

    int lfile = startfile+numrun;

    for(int ifile = startfile; ifile < lfile; ++ifile) {

        for(unsigned int i=0; i<hp.size(); ++i) { 
            if(hp[i] == NULL) continue;
            hp[i]->Reset();
        }
        for(unsigned int i=0; i<profs.size(); i++){
            if(profs[i] == NULL) continue;
            profs[i]->Reset();
        }

        memset(nqbin, 0., sizeof(nqbin));
        memset(qtotal, 0., sizeof(qtotal));
        memset(qsmear, 0., sizeof(qsmear));
        memset(npix, 0., sizeof(npix));
        memset(qflx, 0., sizeof(qflx));
        memset(qfly, 0., sizeof(qfly));
        memset(nelec, 0., sizeof(nelec));
        memset(qbins, 0., sizeof(qbins));
        memset(qbin_merge, 0., sizeof(qbin_merge));
        memset(xwidth, 0, sizeof(xwidth));
        memset(xstart, 0, sizeof(xstart));
        memset(ywidth, 0, sizeof(ywidth));
        memset(ystart, 0, sizeof(ystart));
        memset(good_clust, 0, sizeof(good_clust));
        zero_3d_array(cluster, nevents, TXSIZE, TYSIZE);
        zero_2d_array(xsum1, nevents, TXSIZE);
        zero_2d_array(xsum2, nevents, TXSIZE);
        zero_2d_array(ysum1, nevents, TYSIZE);
        zero_2d_array(ysum2, nevents, TYSIZE);

        for(int i=0; i< n_hists; i++){
            chi_min[i] = 10.;
        }


        //  Read in 1D z template information first

        sprintf(infile,"./ztemp_%5.5d.txt",ifile);

        //  Open input file and read header info 

        FILE *ztemp_file = fopen(infile, "r");
        if (ztemp_file==NULL) {
            printf("no z-template file %s \n", infile);
            return 0;
        }

        fscanf(ztemp_file,"%f  %f  %f", &cosy, &cosx, &cosz);
        //	   printf("cosx/cosy/cosz = %f/%f/%f \n", cosx, cosy, cosz);

        fscanf(ztemp_file,"%f  %f  %f", &qavg_raw, &symax, &pixmaxy);
        printf("qavg_raw/symax/pixmaxy = %f/%f/%f \n", qavg_raw, symax, pixmaxy);

        symaxx = fmax*symax;

        //flip to match cmssw coords
        for(int i = 1; i > -1; --i) {
            fscanf(ztemp_file,"%f %f %f %f %f", &ypar[i][0], &ypar[i][1], &ypar[i][2], &ypar[i][3], &ypar[i][4]);
            printf("Pars are %.4e %.4e %.4e %.4e %.4e \n", ypar[i][0], ypar[i][1], ypar[i][2], ypar[i][3], ypar[i][4]);
        }

        for (int k=0; k < 9; ++k) {

            // Skip labels   
            get_label(ztemp_file, label, 160);
            printf("%d %s\n", k, label);
            //read in template
            for(int i=0; i<TYSIZE; i++){
                fscanf(ztemp_file, " %f ", &ytemp[k][i]);
            }
        }

        fclose(ztemp_file);

        // Calculate the mean cluster size in pixels
        clslny = get_clust_len(ytemp, TYSIZE, symaxx);


        //  Read in 1D p template information

        sprintf(infile,"./ptemp_%5.5d.txt",ifile);


        FILE *ptemp_file = fopen(infile, "r");
        if (ptemp_file==NULL) {
            printf("no p-template file %s \n", infile);
            return 0;
        }

        fscanf(ptemp_file,"%f  %f  %f", &cosy, &cosx, &cosz);
        //	   printf("cosx/cosy/cosz = %f/%f/%f \n", cosx, cosy, cosz);

        fscanf(ptemp_file,"%f  %f  %f", &qavg_raw, &sxmax, &pixmaxx);
        printf("qavg_raw/sxmax/pixmaxx = %f/%f/%f \n", qavg_raw, sxmax, pixmaxx);

        pixmax = std::max(pixmaxx, pixmaxy);

        sxmaxx = fmax*sxmax;

        for(int i = 1; i > -1; --i) {
            fscanf(ptemp_file,"%f %f %f %f %f", &xpar[i][0], &xpar[i][1], &xpar[i][2], &xpar[i][3], &xpar[i][4]);
            printf("Pars are %.4e %.4e %.4e %.4e %.4e \n", xpar[i][0], xpar[i][1], xpar[i][2], xpar[i][3], xpar[i][4]);
        }

        for (int k=0; k < 9; ++k) {
            // Skip labels   
            get_label(ptemp_file, label, 160);
            printf("%s\n", label);
            //read in template
            for(int i=0; i<TXSIZE; i++){
                fscanf(ptemp_file, " %f ", &xtemp[k][i]);
            }
        }
        fclose(ptemp_file);

        // Calculate the mean cluster size in pixels

        clslnx = get_clust_len(xtemp, TXSIZE, sxmaxx);





        //  Open input file and read header info 



        sprintf(infile,"template_events_d%05i_temp.out",ifile);

        printf("opening file %s to get pixel events \n", infile);

        //  Open input file and read header info 

        FILE *events_file = fopen(infile, "r");
        if (events_file==NULL) {
            printf("no pixel data file found: %s \n", infile);
            return 0;
        }

        // Read-in a header string first and print it    

        get_label(events_file, header, 80);

        printf("Header: %s\n", header);
        fscanf(events_file,"%f  %f  %f", &ysize, &xsize, &thick);
        zcen = thick/2.;
        printf("xsize/ysize/thick = %f/%f/%f \n", xsize, ysize, thick);

        float qavg = 0.f; //average charge after threshholding effects
        if(write_temp_header && ifile == startfile) {


            fprintf(temp_output_file,"%s", header);
            fprintf(temp_output_file,"%d %d %4.2f %d %d %d %d %5.4f %5.4f %4.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %f %4.4f %4.4f %4.4f \n",
                    id,nvers,Bfield,NTy,NTyx,NTxx,IDtype,Vbias, temp,fluenc,qscale,q50,lorwdy,
                    lorwdx,ysize,xsize,thick,q51,lorbsy,lorbsx,fbin[0], fbin[1], fbin[2]);

            int ngen_ver = 1;
            fprintf(generr_output_file,"%s", header);
            fprintf(generr_output_file,"%d %d %4.2f %d %d %d %d %5.4f %5.4f %4.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %f %4.4f %4.4f %4.4f \n",
                    id,ngen_ver,Bfield,NTy,NTyx,NTxx,IDtype,Vbias, temp,fluenc,qscale,q50,lorwdy,
                    lorwdx,ysize,xsize,thick,q51,lorbsy,lorbsx,fbin[0], fbin[1], fbin[2]);
        }



        // loop over all events once to get single pixel avgs.
        int read_events = 0;
        qmsort.clear();


        for(int n=0; n<nevents; n++){

            float x1,y1,z1;
            //x and y flipped order in input files
            if(fscanf(events_file,"%f %f %f %f %f %f %d", &y1, &x1, &z1, &cosy, &cosx, &cosz, &nelec[n]) == EOF){
                printf("File %s ended early!! \n\n", infile);
                break;
            }
            read_events++;

            // read the input cluster 
            read_cluster(events_file, pixin);


            cotalpha = cosx/cosz;
            cotbeta = cosy/cosz;

            //  Pixelav gives hit position at face of pixel, translate to
            //  3d center of the pixel
            //  propagate from edge to center of pixel
            //  negative sign to convert to cmssw coords
            //  these hit coords are centered at 0
            xhit[n] = -(x1 + (zcen - z1) * cotalpha);
            yhit[n] = -(y1 + (zcen - z1) * cotbeta);

            int ndcol = TYSIZE/2 +1;
            std::vector<int> ndhit(ndcol, 0);
            int idcol;

            //sigraw is zero padded to allow overflow in double col
            //projections
            memset(sigraw, 0., sizeof(sigraw));
            memset(clust, 0., sizeof(clust));

            // Add noise and analog response to cluster, reformat for flipped barrel coordinate system 
            //

            //saw random numbers used so can use again for double sized version
            float wgraw[TXSIZE][TYSIZE], xgraw[TXSIZE][TYSIZE], ygraw[TXSIZE][TYSIZE], zgraw[TXSIZE][TYSIZE];


            triplg(vgauss);
            qsmear[n] = (1.+vgauss[0]*common_frac);
            pixlst.clear();
            for(int i=0; i<ndcol; ++i) {ndhit[i] = 0;}
            int icol = 0;
            if(vgauss[1] < 0.) {icol = 1;}
            int xtalk_row_start = 0;
            int xtalk_unfold_row = 1;
            if(vgauss[2] < 0.) {
                xtalk_row_start = 1;
                xtalk_unfold_row = 0;
            }

            float xtalk_apply = xtalk_frac + xtalk_noise * vgauss[3];

            if (xtalk_frac > 0.) apply_xtalk(pixin, xtalk_row_start, xtalk_apply);
            
            //do main cluster
            for(int j=0; j<TXSIZE; ++j) {
                triplg(wgauss);
                triplg(xgauss);
                triplg(ygauss);
                triplg(zgauss);
                for(int i=0; i<TYSIZE; ++i) {
                    wgraw[TXSIZE-1-j][TYSIZE-1-i] = wgauss[i];
                    xgraw[TXSIZE-1-j][TYSIZE-1-i] = xgauss[i];
                    ygraw[TXSIZE-1-j][TYSIZE-1-i] = ygauss[i];
                    zgraw[TXSIZE-1-j][TYSIZE-1-i] = zgauss[i];

                    sigraw[TXSIZE-1-j][TYSIZE-1-i] = rten * pixin[j][i];
                    if(rten * pixin[j][i] > 200.) qin = (rten*pixin[j][i] + xgauss[i]*noise);
                    else qin = 0.;
                    rclust[TXSIZE-1-j][TYSIZE-1-i] = qin;
                    if(qin < q100*(1.+wgauss[i]*q100_frac)) {
                        clust[TXSIZE-1-j][TYSIZE-1-i] = 0.;
                    } else {
                        idcol = (TYSIZE-1-i+icol)/2;
                        ndhit[idcol]++;
                        signal = frontEnd.apply_model( qin, ygauss[i], zgauss[i] );
                        clust[TXSIZE-1-j][TYSIZE-1-i] = qsmear[n]*signal;
                    }
                }


            }


            // Simulate the second, higher threshold in single double col hits
            for(int j=0; j<TXSIZE; ++j) {
                for(int i=0; i<TYSIZE; ++i) {
                    if(clust[j][i] > 0.) {
                        idcol = (i+icol)/2;
                        if(ndhit[idcol] == 1) {
                            // Apply higher threshold on single hits in double columns
                            if(rclust[j][i] < q101*(1.+wgauss[i]*q100_frac)) {
                                clust[j][i] = 0.;
                            }
                        }
                    }
                }
            }


            if(xtalk_frac > 0.) unfold_xtalk(clust, xtalk_unfold_row, xtalk_frac);

            // Simulate the seed finding

            qmax = 0.;
            for(int i=0; i<TXSIZE; ++i) {
                for(int j=0; j<TYSIZE; ++j) {
                    if(clust[i][j] > qmax) {
                        qmax = clust[i][j];
                        max.first = i; max.second = j;

                    }
                }
            }

            if(qmax < clustering_thresh){
                good_clust[n] = false;
                continue;
            }
            good_clust[n] = true;


            // Simulate clustering around maximum signal (seed)
            //
            pixlst.clear();
            pixlst.push_back(max);
            memset(bclust, false, sizeof(bclust));
            bclust[max.first][max.second] = true;

            std::vector<std::pair<int, int> > pixlst_copy;

            int numadd = 1;

            //  Iteratively find all non zero pixels near our seed
            while(numadd > 0){
                //  Use copy of vector to avoid modifying vector as we loop through it
                pixlst_copy = pixlst;
                numadd = 0;
                for ( auto pixIter = pixlst_copy.begin(); pixIter != pixlst_copy.end(); ++pixIter ) {
                    //max's are +2 because we are doing <max in the loop
                    int imin = pixIter->first-1; 
                    int imax = pixIter->first+2;
                    int jmin = pixIter->second-1;
                    int jmax = pixIter->second+2;
                    if(imin < 0) {imin = 0;}
                    if(imax > TXSIZE) {imax = TXSIZE;}
                    if(jmin < 0) {jmin = 0;}
                    if(jmax > TYSIZE) {jmax = TYSIZE;}
                    for(int i=imin; i<imax; ++i) {
                        for(int j=jmin; j<jmax; ++j) {
                            if(clust[i][j] > q100) {
                                if(!bclust[i][j]) {
                                    bclust[i][j] = true;
                                    pixel.first = i; pixel.second = j;
                                    pixlst.push_back(pixel);
                                    ++numadd;
                                }
                            }
                        }
                    }
                }
            }

            float qmeas=0.;
            npix[n] = 0.;
            for (auto pixIter = pixlst.begin() ; pixIter != pixlst.end(); ++pixIter ) {
                int i = pixIter->first; 
                int j = pixIter->second;
                qmeas += clust[i][j];
                cluster[n][i][j] = clust[i][j];
                npix[n] += 1.;
            }
            qtotal[n] = qmeas;


            //keep 60 smallest charges
            if(qmsort.size() < 60){
                qmsort.insert(qmeas);
            }
            else if(qmeas < *(qmsort.rbegin())){
                //its smaller than something in the list, remove largest
                //element and this to list
                qmsort.erase(qmsort.find(*qmsort.rbegin()));
                qmsort.insert(qmeas);
            }

            qavg += qmeas;

            //do x double  row projections
            for(int j=0; j<TXSIZE/2; j++){
                xsum1[n][j] = xsum2[n][j] = 0.;
                for(int i=0; i<TYSIZE; i++){
                    int j1 = 2*j;
                    int j2 = 2*j+1;

                    //sigraw padded with extra 0's to allow overflow
                    qin = (sigraw[j1][i] + sigraw[j1+1][i]);
                    qin += xgraw[j1][i]*noise;
                    if(qin > q100*(1.+wgraw[j1][i]*q100_frac)) {
                        signal = frontEnd.apply_model( qin, ygraw[j1][i], zgraw[j1][i] );
                        xsum1[n][j] += qsmear[n]*signal;
                    }

                    qin = (sigraw[j2][i] + sigraw[j2+1][i]);
                    qin += xgraw[j1][i]*noise;
                    if(qin > q100*(1.+wgraw[j2][i]*q100_frac)) {
                        signal = frontEnd.apply_model( qin, ygraw[j2][i], zgraw[j2][i] );
                        xsum2[n][j] += qsmear[n]*signal;
                    }
                }
            }


            //do y double col projections 
            for(int i=0; i<TYSIZE/2; ++i) {
                ysum1[n][i] = ysum2[n][i] = 0.;
                int i1 = 2*i;
                int i2 = 2*i+1;
                for(int j=0; j<TXSIZE; j++){

                    //sigraw padded with extra 0's to allow overflow
                    qin = (sigraw[j][i1] + sigraw[j][i1+1]);
                    qin += xgraw[j][i1]*noise;
                    if(qin > q100*(1.+wgraw[j][i1]*q100_frac)) {
                        signal = frontEnd.apply_model( qin, ygraw[j][i1], zgraw[j][i1] );
                        ysum1[n][i] += qsmear[n]*signal;
                    }

                    //sigraw padded with extra 0's to allow overflow
                    qin = (sigraw[j][i2] + sigraw[j][i2+1]);
                    qin += xgraw[j][i2]*noise;
                    if(qin > q100*(1.+wgraw[j][i2]*q100_frac)) {
                        signal = frontEnd.apply_model( qin, ygraw[j][i2], zgraw[j][i2] );
                        ysum2[n][i] += qsmear[n]*signal;
                    }
                }
            }

        }

        qavg /= read_events;
        fclose(events_file);

        int nxone=0;
        int nyone=0;
        int nxtwo=0;
        int nytwo=0;

        float dxone=0;
        float dyone=0;
        float dxtwo=0;
        float dytwo=0;

        float sxone=0.;
        float syone=0.;
        float sxtwo=0.;
        float sytwo=0.;


        //compute averages over all clusters 
        for(int n=0; n<read_events; n++){
            if(!good_clust[n]) continue;

            float qmeas = qtotal[n];
            float rcorr = qmeas/float(nelec[n]); //ratio of measured charge to generated charge


            hp[charge_idx]->Fill(float(nelec[n]));
            hp[charge_idx + 1] ->Fill(qmeas);
            hp[charge_idx + 7]->Fill(rcorr);

            float qmerge = 0.;
            if(n>0){
                qmerge = qtotal[n] + qsmear[n]*qtotal[n-1]/qsmear[n-1];
                hp[charge_idx+6] ->Fill(qmerge);
            }

            float q_frac = qmeas / qavg;
            if(q_frac > fbin[0]) {
                qbins[n]=0;
            } 
            else if(q_frac > fbin[1]) {
                qbins[n]=1;
            } 
            else if(q_frac > fbin[2]) {
                qbins[n]=2;
            } 
            else {
                qbins[n]=3;
            }
            nqbin[qbins[n]]++;

            float q_frac_merge = qmerge / qavg;

            if(q_frac_merge > 2.* fbin[0]) {
                qbin_merge[n]=0;
            } 
            else if(q_frac > 2.* fbin[1]) {
                qbin_merge[n]=1;
            } 
            else if(q_frac > 2.* fbin[2]) {
                qbin_merge[n]=2;
            } 
            else {
                qbin_merge[n]=3;
            }




            float xsum[TXSIZE], ysum[TYSIZE];

            //x and y projections
            memset(xsum, 0., sizeof(xsum));
            memset(ysum, 0., sizeof(ysum));
            for(int i=0; i<TXSIZE; i++){
                for(int j=0; j<TYSIZE; j++){
                    //smooth cluster charge by applying cap of pixmax
                    //float q = std::min(cluster[n][i][j], pixmax);
                    float q = cluster[n][i][j];
                    xsum[i] += q;
                    ysum[j] += q;
                }
            }

            //get width and start of clusters in 1d projections
            xwidth[n]=0;
            ywidth[n]=0;
            xstart[n]=0;
            ystart[n]=0;

            int xw1(0), xw2(0), yw1(0), yw2(0);
            int xc1(0), xc2(0), yc1(0), yc2(0);

            for(int i=0; i<TXSIZE; i++){
                if(xsum[i] >0.){
                    if(xstart[n]==0) xstart[n] = i;
                    xwidth[n]++;
                }
            }
            for(int j=0; j<TYSIZE; j++){
                if(ysum[j] >0.){
                    if(ystart[n]==0) ystart[n] = j;
                    ywidth[n]++;
                }
            }


            //do double col version
            for(int i=0; i<TXSIZE/2; i++){
                if(xsum1[n][i] >0.){
                    if(xc1 == 0) xc1 = i;
                    xw1++;
                }
                if(xsum2[n][i] >0.){
                    if(xc2 == 0) xc2 = i;
                    xw2++;
                }
            }
            for(int j=0; j<TYSIZE/2; j++){
                if(ysum1[n][j] >0.){
                    if(yc1 == 0) yc1 = j;
                    yw1++;
                }
                if(ysum2[n][j] >0.){
                    if(yc2 == 0) yc2 = j;
                    yw2++;
                }
            }



            //compute front and back signal fractions
            //Fraction of charge loss between front and back
            int xlast = xstart[n] + xwidth[n] -1;
            int ylast = ystart[n] + ywidth[n] -1;

            float xfrac = (xsum[xstart[n]] - xsum[xlast]) / (xsum[xstart[n]] + xsum[xlast]);
            float yfrac = (ysum[ystart[n]] - ysum[ylast]) / (ysum[ystart[n]] + ysum[ylast]);

            qfly[n] = yfrac;
            qflx[n] = xfrac;

            //compute avg shift and variance of single pixel clusters
            if(xwidth[n] ==1){
                nxone++;
                float x0 = (xstart[n] - TXSIZE/2) *xsize;
                float deltax = x0-xhit[n];
                //printf("1pix x: %.1f %.1f \n", x0, xhit[n]);
                dxone += deltax;
                sxone += deltax*deltax;
            }
            if(ywidth[n] ==1){
                nyone++;
                float y0 = (ystart[n] - TYSIZE/2) *ysize;
                float deltay = y0-yhit[n];
                //printf("1pix y: %.1f %.1f \n", y0, yhit[n]);
                dyone += deltay;
                syone += deltay*deltay;
            }

            //do the same for double sized single pixel clusters
            if(xw1 ==1){
                nxtwo++;
                //want middle of our double sized
                float x0 = (xc1*2 + 1 - float(TXSIZE)/2) *xsize;
                float deltax = x0-xhit[n];
                //printf("2pix x: %.1f %.1f \n", x0, yhit[n]);
                dxtwo += deltax;
                sxtwo += deltax*deltax;
            }
            if(xw2 ==1){
                nxtwo++;
                float x0 = (xc2*2 +2 - float(TXSIZE)/2) *xsize;
                float deltax = x0-xhit[n];
                //printf("2pix x: %.1f %.1f \n", x0, yhit[n]);
                dxtwo += deltax;
                sxtwo += deltax*deltax;
            }
            if(yw1 ==1){
                nytwo++;
                //want middle of our double sized
                float y0 = (yc1*2 +1 - float(TYSIZE)/2) *ysize;
                float deltay = y0-yhit[n];
                //printf("2piy y: %.1f %.1f \n", y0, yhit[n]);
                dytwo += deltay;
                sytwo += deltay*deltay;
            }
            if(yw2 ==1){
                nytwo++;
                float y0 = (yc2*2 +2 - float(TYSIZE)/2) *ysize;
                float deltay = y0-yhit[n];
                //printf("2piy y: %.1f %.1f \n", y0, yhit[n]);
                dytwo += deltay;
                sytwo += deltay*deltay;
            }

        }


        //compute avgs and std devs of single pixel residuals if there are enough events
        if(nyone <= 10){
            dyone=lorbsy;
            syone = ysize/sqrt(12);
        }
        else{
            dyone /= float(nyone);
            syone = syone/float(nyone) - dyone*dyone;
            if(syone < 0.) syone = 0.;
            syone = sqrt(syone);
        }

        if(nytwo <= 10){
            dytwo=lorbsy;
            sytwo = 2.*ysize/sqrt(12);
        }
        else{
            dytwo /= float(nytwo);
            sytwo = sytwo/float(nytwo) - dytwo*dytwo;
            if(sytwo < 0.) sytwo = 0.;
            sytwo = sqrt(sytwo);
        }

        if(nxone <= 10){
            dxone=lorbsx;
            sxone =xsize/sqrt(12);
        }
        else{
            dxone /= float(nxone);
            sxone = sxone/float(nxone) - dxone*dxone;
            if(sxone < 0.) sxone = 0.;
            sxone = sqrt(sxone);
        }

        if(nxtwo <= 10){
            dxtwo=lorbsx;
            sxtwo = 2.0*xsize/sqrt(12);
        }
        else{
            dxtwo /= float(nxtwo);
            sxtwo = sxtwo/float(nxtwo) - dxtwo*dxtwo;
            if(sxtwo < 0.) sxtwo = 0.;
            sxtwo = sqrt(sxtwo);
        }

        printf("%i Single pixel y-clusters avg offset: %.1f, std dev %.1f \n", nyone, dyone, syone);
        printf("%i Single big pixel y-clusters  avg offset: %.1f, std dev %.1f \n", nytwo, dytwo, sytwo);
        printf("%i Single pixel x-clusters avg offset: %.1f, std dev %.1f \n", nxone, dxone, sxone);
        printf("%i Single big pixel x-clusters  avg offset: %.1f, std dev %.1f \n", nxtwo, dxtwo, sxtwo);



        // Copy info into the slice and reformat from pixelav coordinates to CMSSW local coordinates
        slice = new SiPixelTemplateEntry;


        slice->runnum = ifile;

        for(int i = 0; i < 2; ++i) {
            for(int j=0; j<5; ++j) {
                slice->xpar[i][j] = xpar[i][j];
                slice->ypar[i][j] = ypar[i][j];
            }
        }

        slice->clslenx = clslnx;
        slice->clsleny = clslny;




        slice->sxone = sxone;
        slice->dxone = dxone;
        slice->sxtwo = sxtwo;
        slice->dxtwo = dxtwo;

        slice->syone = syone;
        slice->dyone = dyone;
        slice->sytwo = sytwo;
        slice->dytwo = dytwo;



        slice->costrk[0] = -cosx;
        slice->costrk[1] = -cosy;
        slice->costrk[2] = -cosz;
        slice->cotalpha = cosx/cosz;
        slice->cotbeta = cosy/cosz;

        printf("qavg/sxmax/pixmax = %f/%f/%f \n", qavg, sxmax, pixmax);

        slice->qavg = qavg;
        slice->pixmax = pixmax;
        slice->sxmax = sxmax;
        slice->symax = symax;

        //fill templates into slice
        for(int k = 0; k < 9; k++){
            //read in templates reversed because of coordinate difference
            //between cmssw and pixelav
            for(int i=0; i<TXSIZE; i++){
                slice->xtemp[9-k-1][TXSIZE - 1 - i] = xtemp[k][i];
            }
            for(int j=0; j<TYSIZE; j++){
                slice->ytemp[9-k-1][TYSIZE - 1 - j] = ytemp[k][j];
            }
        }


        //do first pass of template reco with no charge loss correction
        for(int i=0; i<4; i++){
            for(int j=0; j<6; j++){
                slice->yflpar[i][j] = 0.;
                slice->xflpar[i][j] = 0.;
            }
        }


        locBx = 1.;
        if(cotbeta < 0.) locBx = -1.;
        locBz = locBx;
        if(cotalpha < 0.) locBz = -locBx;

        templ.sideload(slice, IDtype, locBx, locBz, lorwdy, lorwdx, q50, fbin, xsize, ysize, thick);


        nbad = 0;
        ngood = 0;





        // Loop over all clusters and apply generic and first pass of template reco (no charge loss correction) 
        for(int n=0; n<read_events; n++){
            if(!good_clust[n]) continue;


            // Do generic reco on the cluster

            float xsum[TXSIZE], ysum[TYSIZE];

            //print_cluster(cluster[n]);

            memset(xsum, 0., sizeof(xsum));
            memset(ysum, 0., sizeof(ysum));
            for(int i=0; i<TXSIZE; i++){
                for(int j=0; j<TYSIZE; j++){
                    //smooth cluster charge
                    //float q = std::min(cluster[n][i][j], pixmax);
                    float q = cluster[n][i][j];
                    xsum[i] += q;
                    ysum[j] += q;
                }
            }

            int xend = xstart[n] + xwidth[n] -1;
            int yend = ystart[n] + ywidth[n] -1;

            //charges of first and last
            float Q_f_x = xsum[xstart[n]];
            float Q_l_x = xsum[xend];
            float Q_f_y = ysum[ystart[n]];
            float Q_l_y = ysum[yend];

            //edges of cluster
            //
            //f is upper edge of first pixel
            //l is lower edge of last pixel
            float e_f_x  = (xstart[n]+1) *xsize;
            float e_l_x  = (xend)*xsize; 
            float e_f_y  = (ystart[n]+1) *ysize;
            float e_l_y  = (yend) *ysize;


            bool isBigPix = false;

            //taken from CPEGeneric config as of June 2019
            float eff_charge_cut_lowX = 0.;
            float eff_charge_cut_lowY = 0.;
            float eff_charge_cut_highX = 1.0;
            float eff_charge_cut_highY = 1.0;
            float size_cutX = 3.0;
            float size_cutY = 3.0;

            //lorentz widths are passed with a negative sign to get combination
            //with the sign of angle correct (we want to 'add' them) 
// ================================================================================================================
            float xrec_gen = SiPixelUtils::generic_position_formula(xwidth[n], Q_f_x, Q_l_x, e_f_x, e_l_x,
                    -lorwdx, thick, cotalpha,
                    xsize, isBigPix, isBigPix,
                    eff_charge_cut_lowX, eff_charge_cut_highX,
                    size_cutX) - lorbsx;

            float yrec_gen = SiPixelUtils::generic_position_formula(ywidth[n], Q_f_y, Q_l_y, e_f_y, e_l_y,
                    -lorwdy, thick, cotbeta,
                    ysize, isBigPix, isBigPix,
                    eff_charge_cut_lowY, eff_charge_cut_highY,
                    size_cutY) - lorbsy;




            //coordinates returned based origin being top left corner of
            //template
            //hit positions based on center of central pixel in template
            float dx_gen = xrec_gen - (TXSIZE/2.)*xsize - xhit[n];
            float dy_gen = yrec_gen - (TYSIZE/2.)*ysize - yhit[n];


            if(ywidth[n] > 1){
                //printf("Y: dy %.1f Size %i Q %.0f %.0f e %.1f %.1f lordwy %.2f cotbeta %.2f lorbsy %.2f \n",
                    //dy_gen, ywidth[n], Q_f_y, Q_l_y, e_f_y, e_l_y, lorwdy, cotbeta, lorbsy);
                hp[y_generic_idx]->Fill(dy_gen);
                hp[y_generic_idx+1 +qbins[n]]->Fill(dy_gen);
            }

            if(xwidth[n] > 1){
                //printf("X: xgeneric %.1f dx %.1f Size %i Q %.0f %.0f e %.1f %.1f lordwx %.2f cotalpha %.2f lorbsx %.2f \n",
                    //xrec_gen, dx_gen, xwidth[n], Q_f_x, Q_l_x, e_f_x, e_l_x, lorwdx, cotalpha, lorbsx);
                hp[x_generic_idx]->Fill(dx_gen);
                hp[x_generic_idx+1 +qbins[n]]->Fill(dx_gen);
            }



            float cluster_local[TXSIZE][TYSIZE];
            memset(cluster_local, 0., sizeof(cluster_local));
            for(int i=0; i<TXSIZE; i++){
                for(int j=0; j<TYSIZE; j++){
                    cluster_local[i][j] = cluster[n][i][j];
                }
            }
//============================================ NN reco ===============================================================
          
	//printf("\n ===================GOING TO ENTER nn_reco=======================\n ");
          //  do_1dcnn_reco(cluster_local, cotalpha, cotbeta, xrec, yrec);
            angles.tensor<float,2>()(0, 0) = cotalpha;
           angles.tensor<float,2>()(0, 1) = cotbeta;
printf("%s\n","starting x reco");
            for (size_t i = 0; i < TXSIZE; i++) {
            cluster_flat_x.tensor<float,3>()(0, i, 0) = 0;
            for (size_t j = 0; j < TYSIZE; j++){
                //1D projection in x
                cluster_flat_x.tensor<float,3>()(0, i, 0) += cluster_local[i][j];
            }
          }
          printf("%s\n","starting x Run");
          // define the output and run
         // auto start = high_resolution_clock::now();
         status = session_x->Run({{inputTensorName_x,cluster_flat_x}, {anglesTensorName_x,angles}}, {outputTensorName_}, {},&output_x);
         //auto stop = high_resolution_clock::now();
            //printf("Inference time for x = %0.3f us",duration_cast<microseconds>(stop-start));
          // print the output
          //std::cout << "THIS IS THE FROM THE 1DCNN xrec -> " << output_x[0].matrix<float>()(0,0) << std::endl << std::endl;
          xrec = output_x[0].matrix<float>()(0,0);
          printf("%f\n", xrec);

            for (size_t j = 0; j < TYSIZE; j++) {
            cluster_flat_y.tensor<float,3>()(0, j, 0) = 0.;
            for (size_t i = 0; i < TXSIZE; i++){
                //1D projection in x
                cluster_flat_y.tensor<float,3>()(0, j, 0) += cluster_local[i][j];
                
            }
          }
          
          gettimeofday(&t_1dcnn1, &timz);
          // define the output and run
         status = session_y->Run({{inputTensorName_y,cluster_flat_y}, {anglesTensorName_y,angles}}, {outputTensorName_}, {},&output_y);
         gettimeofday(&t_1dcnn2, &timz);
        
        
        
        printf("ellapsed time for 1 cluster = %f seconds \n", t_1dcnn2.tv_usec - t_1dcnn1.tv_usec);
              // print the output
        //std::cout << "THIS IS THE FROM THE 1DCNN yrec -> " << output_y[0].matrix<float>()(0,0) << std::endl << std::endl;
          yrec = output_y[0].matrix<float>()(0,0);

            float dx_1dcnn = xrec - xhit[n];
            float dy_1dcnn = yrec - yhit[n];

             hp[y_1dcnn_idx]->Fill(dy_1dcnn);
             hp[y_1dcnn_idx+1 +qbins[n]]->Fill(dy_1dcnn);
            
             hp[x_1dcnn_idx]->Fill(dx_1dcnn);
             hp[x_1dcnn_idx+1 +qbins[n]]->Fill(dx_1dcnn);
            

//======================================================================================================
          

            }
      
             //grab 30th smallest charge (0.1%)
        auto it = std::next(qmsort.begin(), 29);
        float qmin30 = *it;
        //grab 60th smallest charge (0.2%) 
        it = std::next(qmsort.begin(), 59);
        float qmin60 = *it;


        //output to gen_errors

        fprintf(generr_output_file, "%i %8.6f %8.6f %8.6f \n", ifile, slice->costrk[0], slice->costrk[1], slice->costrk[2]);

        fprintf(generr_output_file, "%8.1f %8.1f %8.1f %8.1f %8.1f %8.1f \n",
                qavg, pixmax, dyone, syone, dxone, sxone);

        fprintf(generr_output_file, "%8.1f %8.1f %8.1f %8.1f %8.1f %8.1f \n",
                dytwo, sytwo, dxtwo, sxtwo, qmin30, qmin60);

        //x and y generic reisduals info
        //USING THE NN RESIDUALS INSTEAD
        for(int i=0; i<4; i++){
            //auto y_pars = get_gaussian_pars(hp[y_generic_idx +1 +i], minErrY);
            //auto x_pars = get_gaussian_pars(hp[x_generic_idx +1 +i], minErrX);
            auto y_pars = get_gaussian_pars(hp[y_1dcnn_idx +1 +i], minErrY);
            auto x_pars = get_gaussian_pars(hp[x_1dcnn_idx +1 +i], minErrX);
            fprintf(generr_output_file, "%9.1f %9.1f %9.1f %9.1f \n", 
                    y_pars[0], y_pars[1], x_pars[0], x_pars[1]);
        }

        // Make plots 
        sprintf(outfile0,"plots/template_histos%5.5d.pdf[",ifile);
        sprintf(outfile1,"plots/template_histos%5.5d.pdf",ifile);
        sprintf(outfile2,"plots/template_histos%5.5d.pdf]",ifile);
        c1->Clear();
        c1->Print(outfile0);
        for(unsigned int i=0; i<hp.size(); ++i) {
            if(hp[i] == NULL) continue;
            hp[i]->Draw();
            c1->Print(outfile1);
            c1->Clear();
        }
        c1->Print(outfile2);
        c1->Clear();

        delete slice;
    }
    // Close output files

    session_x->Close();
    session_y->Close();

    fclose(temp_output_file);  
    fclose(generr_output_file);  

    /*  Determine current time */

    gettimeofday(&now1, &timz);
    deltas = now1.tv_sec - now0.tv_sec;
    deltaus = now1.tv_usec - now0.tv_usec;
    deltat = ((double)deltaus)/1000000.;
    deltat += (double)deltas;
    printf("ellapsed time = %f seconds \n", deltat);

    delete_3d_array(cluster, nevents, TXSIZE, TYSIZE);
    delete_2d_array(xsum1, nevents, TXSIZE);
    delete_2d_array(xsum2, nevents, TXSIZE);
    delete_2d_array(ysum1, nevents, TYSIZE);
    delete_2d_array(ysum2, nevents, TYSIZE);

    return 0;
} // MAIN__ 





