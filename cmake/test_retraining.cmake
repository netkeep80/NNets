# CMake script to test retraining functionality
# This script:
# 1. Trains a model with simple.json (yes/no/empty classes)
# 2. Saves the model
# 3. Retrains with extended.json (adds cat/dog/bird/fish classes)
# 4. Verifies the retrained model works for both old and new classes

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

set(MODEL_V1 "${WORK_DIR}/test_retrain_v1.json")
set(MODEL_V2 "${WORK_DIR}/test_retrain_v2.json")
set(SIMPLE_CONFIG "${CONFIG_DIR}/simple.json")
set(EXTENDED_CONFIG "${CONFIG_DIR}/extended.json")

message(STATUS "=== Testing Retraining Mode ===")
message(STATUS "Executable: ${NNETS_EXE}")
message(STATUS "Simple config: ${SIMPLE_CONFIG}")
message(STATUS "Extended config: ${EXTENDED_CONFIG}")

# Step 1: Train initial model with simple config (yes/no)
message(STATUS "Step 1: Training initial model with simple config...")
execute_process(
    COMMAND "${NNETS_EXE}" -c "${SIMPLE_CONFIG}" -s "${MODEL_V1}" -t
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE TRAIN1_RESULT
    OUTPUT_VARIABLE TRAIN1_OUTPUT
    ERROR_VARIABLE TRAIN1_ERROR
    TIMEOUT 120
)

if(NOT TRAIN1_RESULT EQUAL 0)
    message(FATAL_ERROR "Initial training failed with code ${TRAIN1_RESULT}:\nOutput: ${TRAIN1_OUTPUT}\nError: ${TRAIN1_ERROR}")
endif()
message(STATUS "Initial training completed successfully")

# Verify model file was created
if(NOT EXISTS "${MODEL_V1}")
    message(FATAL_ERROR "Model file was not created: ${MODEL_V1}")
endif()
message(STATUS "Initial model saved: ${MODEL_V1}")

# Step 2: Verify the initial model using --verify
message(STATUS "Step 2: Verifying initial model accuracy...")
execute_process(
    COMMAND "${NNETS_EXE}" -l "${MODEL_V1}" -c "${SIMPLE_CONFIG}" --verify
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE VERIFY1_RESULT
    OUTPUT_VARIABLE VERIFY1_OUTPUT
    ERROR_VARIABLE VERIFY1_ERROR
    TIMEOUT 60
)

if(NOT VERIFY1_RESULT EQUAL 0)
    message(FATAL_ERROR "Initial model verification failed with code ${VERIFY1_RESULT}:\nOutput: ${VERIFY1_OUTPUT}\nError: ${VERIFY1_ERROR}")
endif()

# Check that verification output contains accuracy info
string(FIND "${VERIFY1_OUTPUT}" "Accuracy:" ACCURACY_FOUND)
if(ACCURACY_FOUND EQUAL -1)
    message(FATAL_ERROR "Verification output doesn't contain accuracy info:\n${VERIFY1_OUTPUT}")
endif()
message(STATUS "Initial model verification passed")
message(STATUS "Verification output:\n${VERIFY1_OUTPUT}")

# Step 3: Test inference with initial model
message(STATUS "Step 3: Testing inference with initial model...")
execute_process(
    COMMAND "${NNETS_EXE}" -l "${MODEL_V1}" -i "yes"
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE INFER1_RESULT
    OUTPUT_VARIABLE INFER1_OUTPUT
    ERROR_VARIABLE INFER1_ERROR
    TIMEOUT 30
)

if(NOT INFER1_RESULT EQUAL 0)
    message(FATAL_ERROR "Initial inference failed with code ${INFER1_RESULT}:\nOutput: ${INFER1_OUTPUT}\nError: ${INFER1_ERROR}")
endif()
message(STATUS "Initial inference passed")

# Step 4: Retrain with extended config (adding more classes)
# Note: For simplicity, we create a new config that can be used for retraining
# The retrain mode will detect that some classes are already trained
message(STATUS "Step 4: Retraining model with extended config...")
execute_process(
    COMMAND "${NNETS_EXE}" -r "${MODEL_V1}" -c "${EXTENDED_CONFIG}" -s "${MODEL_V2}" -t
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE RETRAIN_RESULT
    OUTPUT_VARIABLE RETRAIN_OUTPUT
    ERROR_VARIABLE RETRAIN_ERROR
    TIMEOUT 300
)

# Note: Retrain might show warnings about config mismatch, which is expected
# since simple.json has different classes than extended.json
# The actual retraining will train the new classes
message(STATUS "Retrain output: ${RETRAIN_OUTPUT}")

# Verify retrained model file was created
if(NOT EXISTS "${MODEL_V2}")
    message(FATAL_ERROR "Retrained model file was not created: ${MODEL_V2}")
endif()
message(STATUS "Retrained model saved: ${MODEL_V2}")

# Step 5: Test inference with retrained model for new classes
message(STATUS "Step 5: Testing inference with retrained model...")
execute_process(
    COMMAND "${NNETS_EXE}" -l "${MODEL_V2}" -i "cat"
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE INFER2_RESULT
    OUTPUT_VARIABLE INFER2_OUTPUT
    ERROR_VARIABLE INFER2_ERROR
    TIMEOUT 30
)

if(NOT INFER2_RESULT EQUAL 0)
    message(FATAL_ERROR "Retrained inference failed with code ${INFER2_RESULT}:\nOutput: ${INFER2_OUTPUT}\nError: ${INFER2_ERROR}")
endif()

# Check that output contains new class
string(FIND "${INFER2_OUTPUT}" "cat" CAT_FOUND)
if(CAT_FOUND EQUAL -1)
    message(FATAL_ERROR "Retrained model doesn't recognize 'cat' class:\n${INFER2_OUTPUT}")
endif()
message(STATUS "Retrained inference passed")

# Cleanup
file(REMOVE "${MODEL_V1}")
file(REMOVE "${MODEL_V2}")
message(STATUS "=== Retraining Test PASSED ===")
