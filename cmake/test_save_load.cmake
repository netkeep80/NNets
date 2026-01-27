# CMake script to test save/load functionality
# This script trains a model, saves it, then tests inference with the saved model

# Check required variables
if(NOT DEFINED NNETS_EXE)
    message(FATAL_ERROR "NNETS_EXE not defined")
endif()

if(NOT DEFINED CONFIG_DIR)
    message(FATAL_ERROR "CONFIG_DIR not defined")
endif()

if(NOT DEFINED WORK_DIR)
    message(FATAL_ERROR "WORK_DIR not defined")
endif()

set(MODEL_FILE "${WORK_DIR}/test_saveload_model.json")
set(CONFIG_FILE "${CONFIG_DIR}/simple.json")

message(STATUS "=== Testing Model Save/Load ===")
message(STATUS "Executable: ${NNETS_EXE}")
message(STATUS "Config: ${CONFIG_FILE}")
message(STATUS "Model output: ${MODEL_FILE}")

# Step 1: Train and save model
message(STATUS "Step 1: Training and saving model...")
execute_process(
    COMMAND "${NNETS_EXE}" -c "${CONFIG_FILE}" -s "${MODEL_FILE}" -t
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE TRAIN_RESULT
    OUTPUT_VARIABLE TRAIN_OUTPUT
    ERROR_VARIABLE TRAIN_ERROR
    TIMEOUT 120
)

if(NOT TRAIN_RESULT EQUAL 0)
    message(FATAL_ERROR "Training failed with code ${TRAIN_RESULT}:\nOutput: ${TRAIN_OUTPUT}\nError: ${TRAIN_ERROR}")
endif()
message(STATUS "Training completed successfully")

# Verify model file was created
if(NOT EXISTS "${MODEL_FILE}")
    message(FATAL_ERROR "Model file was not created: ${MODEL_FILE}")
endif()
message(STATUS "Model file created: ${MODEL_FILE}")

# Step 2: Test inference with "yes" input
message(STATUS "Step 2: Testing inference with 'yes' input...")
execute_process(
    COMMAND "${NNETS_EXE}" -l "${MODEL_FILE}" -i "yes"
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE INFER1_RESULT
    OUTPUT_VARIABLE INFER1_OUTPUT
    ERROR_VARIABLE INFER1_ERROR
    TIMEOUT 30
)

if(NOT INFER1_RESULT EQUAL 0)
    message(FATAL_ERROR "Inference for 'yes' failed with code ${INFER1_RESULT}:\nOutput: ${INFER1_OUTPUT}\nError: ${INFER1_ERROR}")
endif()

# Check that output contains "yes" class
string(FIND "${INFER1_OUTPUT}" "yes" YES_FOUND)
if(YES_FOUND EQUAL -1)
    message(FATAL_ERROR "Inference output doesn't contain 'yes' class:\n${INFER1_OUTPUT}")
endif()
message(STATUS "Inference for 'yes' passed")

# Step 3: Test inference with "no" input
message(STATUS "Step 3: Testing inference with 'no' input...")
execute_process(
    COMMAND "${NNETS_EXE}" -l "${MODEL_FILE}" -i "no"
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE INFER2_RESULT
    OUTPUT_VARIABLE INFER2_OUTPUT
    ERROR_VARIABLE INFER2_ERROR
    TIMEOUT 30
)

if(NOT INFER2_RESULT EQUAL 0)
    message(FATAL_ERROR "Inference for 'no' failed with code ${INFER2_RESULT}:\nOutput: ${INFER2_OUTPUT}\nError: ${INFER2_ERROR}")
endif()

# Check that output contains "no" class
string(FIND "${INFER2_OUTPUT}" "no" NO_FOUND)
if(NO_FOUND EQUAL -1)
    message(FATAL_ERROR "Inference output doesn't contain 'no' class:\n${INFER2_OUTPUT}")
endif()
message(STATUS "Inference for 'no' passed")

# Cleanup
file(REMOVE "${MODEL_FILE}")
message(STATUS "=== Save/Load Test PASSED ===")
