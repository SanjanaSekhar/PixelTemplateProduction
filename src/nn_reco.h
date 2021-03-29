// Functions to perform inference with NN models
// Author: Sanjana Sekhar
// Date: 3/8/21

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

void do_1dcnn_reco(float cluster[TXSIZE][TYSIZE], float cotalpha, float cotbeta, float& xrec, float& yrec)
{
   char *graph_ext = "1dcnn_p1_mar9";
   char graph_x[100],graph_y[100], inputTensorName_[100],outputTensorName_[100];
   sprintf(graph_x,"data/graph_x_%s.pb",graph_ext);
   sprintf(graph_y,"data/graph_y_%s.pb",graph_ext) ;

   //printf("TXSIZE = %i\n", TXSIZE);
   //printf("TYSIZE = %i\n", TYSIZE);
   
   sprintf(inputTensorName_,"input_1");
   sprintf(outputTensorName_,"Identity"); 

   GraphDef graphDef_x;
  Session* session_x;
  Status status; SessionOptions sessionOptions;
 status = NewSession(sessionOptions, &session_x);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    //printf('Session x not okay');
  }
  GraphDef graphDef_y;
  Session* session_y;
   status = NewSession(sessionOptions, &session_y);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    //printf('Session y not okay');
  }
  std::vector<tensorflow::Tensor> output_x;
  std::vector<tensorflow::Tensor> output_y;

  //setLogging("2");
  //=========== infer x ====================
  // load the graph
   status = ReadBinaryProto(Env::Default(), graph_x, &graphDef_x);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
   // printf('Error in graph_x pb');
  }
   // create a new session and add the graphDef
  status = session_x->Create(graphDef_x);
   // define a tensor and fill it with cluster projection
  tensorflow::Tensor input_x(tensorflow::DT_FLOAT, {1,TXSIZE+2,1});
  for (size_t i = 0; i < TXSIZE; i++) {
  	input_x.tensor<float,3>()(0, i, 0) = 0;
  	for (size_t j = 0; j < TYSIZE; j++){
  		//1D projection in x
  		input_x.tensor<float,3>()(0, i, 0) += cluster[i][j];
  	}
  }
  input_x.tensor<float,3>()(0, TXSIZE, 0) = cotalpha;
  input_x.tensor<float,3>()(0, TXSIZE+1, 0) = cotbeta;
  // define the output and run
  auto start = high_resolution_clock::now();
 status = session_x->Run({{inputTensorName_, input_x}}, {outputTensorName_}, {},&output_x);
auto stop = high_resolution_clock::now();
	//printf("Inference time for x = %0.3f us",duration_cast<microseconds>(stop-start));
  // print the output
  //std::cout << "THIS IS THE FROM THE 1DCNN xrec -> " << output_x[0].matrix<float>()(0,0) << std::endl << std::endl;
  xrec = output_x[0].matrix<float>()(0,0);

  session_x->Close();
  //delete graphDef_x;
  //graphDef_x = nullptr;

  //=========== infer y ====================
  // load the graph
   status = ReadBinaryProto(Env::Default(), graph_y, &graphDef_y);

  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
   // printf("Error in reading graph_y");
  }
   // create a new session and add the graphDef
  status = session_y->Create(graphDef_y);
   // define a tensor and fill it with cluster projection
  tensorflow::Tensor input_y(tensorflow::DT_FLOAT, {1,TYSIZE+2,1});
  for (size_t j = 0; j < TYSIZE; j++) {
  	input_y.tensor<float,3>()(0, j, 0) = 0.;
  	for (size_t i = 0; i < TXSIZE; i++){
  		//1D projection in x
  		input_y.tensor<float,3>()(0, j, 0) += cluster[i][j];
		//printf("j = %i, input_y = %0.3f\n",j,input_y.tensor<float,3>()(0, j, 0));
  	}
  }
  input_y.tensor<float,3>()(0, TYSIZE, 0) = cotalpha;
  input_y.tensor<float,3>()(0, TYSIZE+1, 0) = cotbeta;

  // define the output and run
 status = session_y->Run({{inputTensorName_, input_y}}, {outputTensorName_}, {},&output_y);

  // print the output
//std::cout << "THIS IS THE FROM THE 1DCNN yrec -> " << output_y[0].matrix<float>()(0,0) << std::endl << std::endl;
  yrec = output_y[0].matrix<float>()(0,0);

  session_y->Close();
  //delete graphDef_y;
  //graphDef_y = nullptr;

//  return 1;
}
