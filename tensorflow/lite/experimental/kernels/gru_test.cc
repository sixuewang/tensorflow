/* Copyright 2019 Mobvoi Inc. All Rights Reserved.
   Author: sxwang@mobvoi.com (Sixue Wang)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <vector>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace ops {
namespace experimental {

TfLiteRegistration* Register_GRU();

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

class GRUOpModel : public SingleOpModel {
 public:
  GRUOpModel(const std::vector<std::vector<int>>& input_shapes,
             const TensorType& weight_type = TensorType_FLOAT32) {
    input_ = AddInput(TensorType_FLOAT32);
    state_ =
        AddInput(TensorData{TensorType_FLOAT32, {n_batch_, n_output_}}, true);
    gate_weight_ = AddInput(TensorType_FLOAT32);
    gate_bias_ = AddInput(TensorType_FLOAT32);
    candidate_weight_ = AddInput(TensorType_FLOAT32);
    candidate_bias_ = AddInput(TensorType_FLOAT32);

    output_ = AddOutput(TensorType_FLOAT32);
    activation_ = AddOutput(TensorType_FLOAT32);
    concat_ = AddOutput(TensorType_FLOAT32);

    SetCustomOp("GRU", {}, Register_GRU);
    BuildInterpreter(input_shapes);
  }

  void SetInput(const std::vector<float>& f) { PopulateTensor(input_, f); }

  void SetState(const std::vector<float>& f) { PopulateTensor(state_, f); }

  void SetGateWeight(const std::vector<float>& f) {
    PopulateTensor(gate_weight_, f);
  }

  void SetGateBias(const std::vector<float>& f) {
    PopulateTensor(gate_bias_, f);
  }

  void SetCandidateWeight(const std::vector<float>& f) {
    PopulateTensor(candidate_weight_, f);
  }

  void SetCandidateBias(const std::vector<float>& f) {
    PopulateTensor(candidate_bias_, f);
  }

  std::vector<int> GetOutputShape() { return GetTensorShape(output_); }

  std::vector<float> GetOutput() { return ExtractVector<float>(output_); }

  std::vector<float> GetState() { return ExtractVector<float>(state_); }

  int num_batches() { return n_batch_; }
  int num_inputs() { return n_input_; }
  int num_outputs() { return n_output_; }

 protected:
  int input_;
  int state_;
  int gate_weight_;
  int gate_bias_;
  int candidate_weight_;
  int candidate_bias_;

  int output_;
  int activation_;
  int concat_;

  int n_batch_;
  int n_input_;
  int n_output_;
};

TEST(GRUTest, SimpleTest) {
  const int n_batch = 2;
  const int n_input = 2;
  const int n_output = 3;

  GRUOpModel m({{n_batch, n_input},
                {n_batch, n_output},
                {2 * n_output, n_input + n_output},
                {2 * n_output},
                {n_output, n_input + n_output},
                {n_output}});
  m.SetInput({0.89495724, 0.34482682, 0.68505806, 0.7135783});
  m.SetState(
      {0.09992421, 0.3028481, 0.78305984, 0.50438094, 0.11269058, 0.10244724});
  m.SetGateWeight(
      {0.35144985,  0.60738707, 0.020656768, 0.7130033,   0.010830959,
       0.5090408,   0.9196754,  0.42860362,  0.041295025, 0.8898636,
       0.37174186,  0.12528315, 0.92256004,  0.3334856,   0.67933226,
       0.071062885, 0.49279493, 0.38044137,  0.59205,     0.71381366,
       0.84015673,  0.2102545,  0.038757455, 0.72581637,  0.6293595,
       0.15700719,  0.61266047, 0.5176311,   0.3431678,   0.024911262});
  m.SetGateBias(
      {0.98199004, 0.21527472, 0.91729677, 0.049088843, 0.7720907, 0.6777621});
  m.SetCandidateWeight({0.8472328, 0.33588117, 0.9647169, 0.87238103,
                        0.10335992, 0.023149788, 0.4888804, 0.22516328,
                        0.010163158, 0.50070626, 0.08683334, 0.29994115,
                        0.102154665, 0.017968405, 0.14797808});
  m.SetCandidateBias({0.8074161, 0.48580337, 0.64837664});

  m.Invoke();

  EXPECT_THAT(m.GetState(), ElementsAreArray(ArrayFloatNear(m.GetOutput())));
  EXPECT_THAT(m.GetOutputShape(), ElementsAre(n_batch, n_output));
  EXPECT_THAT(
      m.GetOutput(),
      ElementsAreArray(ArrayFloatNear({0.3234734, 0.34573013, 0.77256536,
                                       0.6513261, 0.21554601, 0.21878791})));
}

}  // namespace
}  // namespace experimental
}  // namespace ops
}  // namespace tflite

int main(int argc, char** argv) {
  ::tflite::LogToStderr();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
