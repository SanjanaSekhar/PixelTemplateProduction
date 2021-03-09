// Functions to perform inference with NN models
// Author: Sanjana Sekhar
// Date: 3/8/21

#include "template_utils.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include <memory>

int 1dcnn_reco(float cluster[TXSIZE][TYSIZE] float cotalpha, float cotbeta, float& xrec, float& yrec)
{
   char *graph_ext = "1dcnn_p1_mar9"
   sprintf(graph_x,"data/graph_x_%s.pb",graph_ext)
   sprintf(graph_y,"data/graph_y_%s.pb",graph_ext) 

   printf("TXSIZE = %i\n", TXSIZE);
   printf("TYSIZE = %i\n", TXSIZE);
   
   sprintf(inputTensorName_,"input_1");
   sprintf(outputTensorName_,"Identity"); 

   tensorflow::GraphDef* graphDef_x;
  tensorflow::Session* session_x;
  tensorflow::GraphDef* graphDef_y;
  tensorflow::Session* session_y;
  std::vector<tensorflow::Tensor> output_x;
  std::vector<tensorflow::Tensor> output_y;

  tensorflow::setLogging("2");
  //=========== infer x ====================
  // load the graph
   graphDef_x = tensorflow::loadGraphDef(graph_x);
   // create a new session and add the graphDef
  session_x = tensorflow::createSession(graphDef_x);
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
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1,TYSIZE+2,1});
  for (size_t j = 0; j < TYSIZE; j++) {
  	input.tensor<float,3>()(0, j, 0) = 0.;
  	for (size_t i = 0; i < TXSIZE; i++){
  		//1D projection in x
  		input.tensor<float,3>()(0, j, 0) += cluster[i][j];
  	}
  }
  input.tensor<float,3>()(0, TYSIZE, 0) = cotalpha;
  input.tensor<float,3>()(0, TYSIZE+1, 0) = cotbeta;

  // define the output and run
  tensorflow::run(session_y, {{inputTensorName_, input}}, {outputTensorName_}, &output_y);

  // print the output
  std::cout << "THIS IS THE FROM THE 1DCNN yrec -> " << output_y[0].matrix<float>()(0,0) << std::endl << std::endl;
  yrec = output_y[0].matrix<float>()(0,0);

  tensorflow::closeSession(session_y);
  delete graphDef_y;
  graphDef_y = nullptr;

  return 1;
}