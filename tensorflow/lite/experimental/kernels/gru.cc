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
#include <limits>
#include <vector>

#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace experimental {
namespace gru {

using reference_ops::Concatenation;
using optimized_ops::FullyConnected;
using optimized_ops::ArrayMap;
using optimized_ops::MapAsArrayWithLastDimAsRows;

void GruCell(const RuntimeShape& input_shape, const float* input,
             const RuntimeShape& state_shape, float* state,
             const RuntimeShape& gate_weight_shape, const float* gate_weight,
             const RuntimeShape& gate_bias_shape, const float* gate_bias,
             const RuntimeShape& candidate_weight_shape,
             const float* candidate_weight,
             const RuntimeShape& candidate_bias_shape,
             const float* candidate_bias, const RuntimeShape& output_shape,
             float* output, const RuntimeShape& activation_shape,
             float* activation, const RuntimeShape& concat_shape,
             float* concat) {
  const int n_batch = input_shape.Dims(0);
  const int n_input = input_shape.Dims(1);
  const int n_output = state_shape.Dims(1);

  // [x h] = concat(input, state)
  std::vector<float const*> concat_arrays_data;
  std::vector<RuntimeShape const*> concat_arrays_shapes;
  concat_arrays_data.push_back(input);
  concat_arrays_data.push_back(state);
  concat_arrays_shapes.push_back(&input_shape);
  concat_arrays_shapes.push_back(&state_shape);
  tflite::ConcatenationParams concat_params;
  concat_params.axis = 1;
  concat_params.inputs_count = concat_arrays_data.size();
  Concatenation(concat_params, &(concat_arrays_shapes[0]),
                &(concat_arrays_data[0]), concat_shape, concat);

  // [r u] = [x h] * gate_weight + gate_bias
  tflite::FullyConnectedParams fc_params;
  fc_params.float_activation_min = std::numeric_limits<float>::lowest();
  fc_params.float_activation_max = std::numeric_limits<float>::max();
  FullyConnected(fc_params, concat_shape, concat, gate_weight_shape,
                 gate_weight, gate_bias_shape, gate_bias, activation_shape,
                 activation);

  // [r u] = sigmoid([r u])
  ArrayMap<float> ru =
      MapAsArrayWithLastDimAsRows(activation, activation_shape);
  ru = ru.unaryExpr(Eigen::internal::scalar_logistic_op<float>());
  auto r = ru.block(0 * n_output, 0, n_output, n_batch);
  auto u = ru.block(1 * n_output, 0, n_output, n_batch);

  // hr = h .* r
  ArrayMap<float> h = MapAsArrayWithLastDimAsRows(state, state_shape);
  ArrayMap<float> xh = MapAsArrayWithLastDimAsRows(concat, concat_shape);
  auto hr = xh.block(n_input, 0, n_output, n_batch);
  hr = h * r;

  // c = [x hr] * candidate_weight + candidate_bias
  FullyConnected(fc_params, concat_shape, concat, candidate_weight_shape,
                 candidate_weight, candidate_bias_shape, candidate_bias,
                 output_shape, output);

  ArrayMap<float> c = MapAsArrayWithLastDimAsRows(output, output_shape);
  // output = (1 - u) .* tanh(c) + u .* h
  c = (1.0 - u) * c.tanh() + u * h;

  memcpy(state, output, n_batch * n_output * sizeof(float));
}

enum InputTensor {
  kInput = 0,
  kState = 1,
  kGateWeight = 2,
  kGateBias = 3,
  kCandidateWeight = 4,
  kCandidateBias = 5,
  kInputNum = 6
};

enum OutputTensor { kOutput = 0, kActivation = 1, kConcat = 2, kOutputNum = 3 };

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, kInputNum);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, kOutputNum);

  // input's dim = [n_batch, n_input]
  const TfLiteTensor* input = GetInput(context, node, kInput);
  TF_LITE_ENSURE_EQ(context, input->dims->size, 2);
  const int n_batch = input->dims->data[0];
  const int n_input = input->dims->data[1];

  // state's dim = [n_batch, n_output]
  TfLiteTensor* state = &context->tensors[node->inputs->data[kState]];
  TF_LITE_ENSURE_EQ(context, state->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, state->dims->data[0], n_batch);
  const int n_output = state->dims->data[1];
  state->allocation_type = kTfLiteArenaRwPersistent;

  // gate_weight' dim = [2 * n_output, n_input + n_output]
  const TfLiteTensor* gate_weight = GetInput(context, node, kGateWeight);
  TF_LITE_ENSURE_EQ(context, gate_weight->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, gate_weight->dims->data[0], 2 * n_output);
  TF_LITE_ENSURE_EQ(context, gate_weight->dims->data[1], n_input + n_output);

  // gate_bias' dim = [2 * n_output]
  const TfLiteTensor* gate_bias = GetInput(context, node, kGateBias);
  TF_LITE_ENSURE_EQ(context, gate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, gate_bias->dims->data[0], 2 * n_output);

  // candidate_weight' dim = [n_output, n_input + n_output]
  const TfLiteTensor* candidate_weight =
      GetInput(context, node, kCandidateWeight);
  TF_LITE_ENSURE_EQ(context, candidate_weight->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, candidate_weight->dims->data[0], n_output);
  TF_LITE_ENSURE_EQ(context, candidate_weight->dims->data[1],
                    n_input + n_output);

  // candidate_bias' dim = [n_output]
  const TfLiteTensor* candidate_bias = GetInput(context, node, kCandidateBias);
  TF_LITE_ENSURE_EQ(context, candidate_bias->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, candidate_bias->dims->data[0], n_output);

  // output's dim = [n_batch, n_output]
  TfLiteTensor* output = GetOutput(context, node, kOutput);
  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, output, TfLiteIntArrayCopy(state->dims)));

  // activation's dim = [n_batch, 2 * n_output]
  TfLiteTensor* activation = GetOutput(context, node, kActivation);
  TfLiteIntArray* activation_size = TfLiteIntArrayCreate(2);
  activation_size->data[0] = n_batch;
  activation_size->data[1] = 2 * n_output;
  TF_LITE_ENSURE_OK(
      context, context->ResizeTensor(context, activation, activation_size));

  // concat's dim  = [n_batch, n_input + n_output]
  TfLiteTensor* concat = GetOutput(context, node, kConcat);
  TfLiteIntArray* concat_size = TfLiteIntArrayCreate(2);
  concat_size->data[0] = n_batch;
  concat_size->data[1] = n_input + n_output;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, concat, concat_size));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInput);
  const TfLiteTensor* state = GetInput(context, node, kState);
  const TfLiteTensor* gate_weight = GetInput(context, node, kGateWeight);
  const TfLiteTensor* gate_bias = GetInput(context, node, kGateBias);
  const TfLiteTensor* candidate_weight =
      GetInput(context, node, kCandidateWeight);
  const TfLiteTensor* candidate_bias = GetInput(context, node, kCandidateBias);
  TfLiteTensor* output = GetOutput(context, node, kOutput);
  TfLiteTensor* activation = GetOutput(context, node, kActivation);
  TfLiteTensor* concat = GetOutput(context, node, kConcat);

  if (input->type == kTfLiteFloat32 && state->type == kTfLiteFloat32 &&
      gate_weight->type == kTfLiteFloat32 &&
      gate_bias->type == kTfLiteFloat32 &&
      candidate_weight->type == kTfLiteFloat32 &&
      candidate_bias->type == kTfLiteFloat32 &&
      output->type == kTfLiteFloat32 && activation->type == kTfLiteFloat32) {
    GruCell(GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(state), reinterpret_cast<float*>(state->data.raw),
            GetTensorShape(gate_weight), GetTensorData<float>(gate_weight),
            GetTensorShape(gate_bias), GetTensorData<float>(gate_bias),
            GetTensorShape(candidate_weight),
            GetTensorData<float>(candidate_weight),
            GetTensorShape(candidate_bias),
            GetTensorData<float>(candidate_bias), GetTensorShape(output),
            GetTensorData<float>(output), GetTensorShape(activation),
            GetTensorData<float>(activation), GetTensorShape(concat),
            GetTensorData<float>(concat));
  } else {
    context->ReportError(context,
                         "Unsupported combination of data types for GruCell");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace gru

TfLiteRegistration* Register_GRU() {
  static TfLiteRegistration r = {nullptr, nullptr, gru::Prepare, gru::Eval};
  return &r;
}

}  // namespace experimental
}  // namespace ops
}  // namespace tflite
