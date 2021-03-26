// Functions to perform inference with NN models
// Author: Sanjana Sekhar
// Date: 3/8/21

//#include "template_utils.h"
//#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <memory>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_bundle/naming.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/cc/saved_model/tag_constants.h"

using namespace tensorflow;

void do_1dcnn_reco(float cluster[TXSIZE][TYSIZE], float cotalpha, float cotbeta, float& xrec, float& yrec)
{
   char *graph_ext = "1dcnn_p1_mar9";
   char graph_x[100],graph_y[100], inputTensorName_[100],outputTensorName_[100];
   sprintf(graph_x,"data/graph_x_%s.pb",graph_ext);
   sprintf(graph_y,"data/graph_y_%s.pb",graph_ext) ;

   printf("TXSIZE = %i\n", TXSIZE);
   printf("TYSIZE = %i\n", TXSIZE);
   
   sprintf(inputTensorName_,"input_1");
   sprintf(outputTensorName_,"Identity"); 

   GraphDef* graphDef_x;
  Session* session_x;
  GraphDef* graphDef_y;
  Session* session_y;
  std::vector<tensorflow::Tensor> output_x;
  std::vector<tensorflow::Tensor> output_y;

  setLogging("2");
  //=========== infer x ====================
  // load the graph
   graphDef_x = loadGraphDef(graph_x);
   // create a new session and add the graphDef
  session_x = createSession(graphDef_x);
   // define a tensor and fill it with cluster projection
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1,TXSIZE+2,1});
  for (size_t i = 0; i < TXSIZE; i++) {
  	input.tensor<float,3>()(0, i, 0) = 0;
  	for (size_t j = 0; j < TYSIZE; j++){
  		//1D projection in x
  		input.tensor<float,3>()(0, i, 0) += cluster[i][j];
  	}
  }
  input.tensor<float,3>()(0, TXSIZE, 0) = cotalpha;
  input.tensor<float,3>()(0, TXSIZE+1, 0) = cotbeta;
  // define the output and run
  
  tensorflow::run(session_x, {{inputTensorName_, input}}, {outputTensorName_}, &output_x);

  // print the output
  std::cout << "THIS IS THE FROM THE 1DCNN xrec -> " << output_x[0].matrix<float>()(0,0) << std::endl << std::endl;
  xrec = output_x[0].matrix<float>()(0,0);

  tensorflow::closeSession(session_x);
  delete graphDef_x;
  graphDef_x = nullptr;

  //=========== infer y ====================
  // load the graph
   graphDef_y = tensorflow::loadGraphDef(graph_y);
   // create a new session and add the graphDef
  session_y = tensorflow::createSession(graphDef_y);
   // define a tensor and fill it with cluster projection
  tensorflow::Tensor input_y(tensorflow::DT_FLOAT, {1,TYSIZE+2,1});
  for (size_t j = 0; j < TYSIZE; j++) {
  	input_y.tensor<float,3>()(0, j, 0) = 0.;
  	for (size_t i = 0; i < TXSIZE; i++){
  		//1D projection in x
  		input_y.tensor<float,3>()(0, j, 0) += cluster[i][j];
  	}
  }
  input_y.tensor<float,3>()(0, TYSIZE, 0) = cotalpha;
  input_y.tensor<float,3>()(0, TYSIZE+1, 0) = cotbeta;

  // define the output and run
  tensorflow::run(session_y, {{inputTensorName_, input_y}}, {outputTensorName_}, &output_y);

  // print the output
  std::cout << "THIS IS THE FROM THE 1DCNN yrec -> " << output_y[0].matrix<float>()(0,0) << std::endl << std::endl;
  yrec = output_y[0].matrix<float>()(0,0);

  tensorflow::closeSession(session_y);
  delete graphDef_y;
  graphDef_y = nullptr;

//  return 1;
}
