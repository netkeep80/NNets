# CMake script to test verification mode (--verify)
# This script trains a model and then verifies its accuracy

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

set(MODEL_FILE "${WORK_DIR}/test_verify_model.json")
set(CONFIG_FILE "${CONFIG_DIR}/simple.json")

message(STATUS "=== Testing Verification Mode ===")
message(STATUS "Executable: ${NNETS_EXE}")
message(STATUS "Config: ${CONFIG_FILE}")
message(STATUS "Model: ${MODEL_FILE}")

# Step 1: Train and save model
message(STATUS "Step 1: Training model...")
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

# Step 2: Verify model accuracy using --verify
message(STATUS "Step 2: Verifying model accuracy...")
execute_process(
    COMMAND "${NNETS_EXE}" -l "${MODEL_FILE}" -c "${CONFIG_FILE}" --verify
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE VERIFY_RESULT
    OUTPUT_VARIABLE VERIFY_OUTPUT
    ERROR_VARIABLE VERIFY_ERROR
    TIMEOUT 60
)

if(NOT VERIFY_RESULT EQUAL 0)
    message(FATAL_ERROR "Verification failed with code ${VERIFY_RESULT}:\nOutput: ${VERIFY_OUTPUT}\nError: ${VERIFY_ERROR}")
endif()

# Check that verification output contains expected elements
string(FIND "${VERIFY_OUTPUT}" "Verifying model accuracy" VERIFY_HEADER)
if(VERIFY_HEADER EQUAL -1)
    message(FATAL_ERROR "Verification output missing header:\n${VERIFY_OUTPUT}")
endif()

string(FIND "${VERIFY_OUTPUT}" "Accuracy:" ACCURACY_FOUND)
if(ACCURACY_FOUND EQUAL -1)
    message(FATAL_ERROR "Verification output missing accuracy:\n${VERIFY_OUTPUT}")
endif()

string(FIND "${VERIFY_OUTPUT}" "Passed:" PASSED_FOUND)
if(PASSED_FOUND EQUAL -1)
    message(FATAL_ERROR "Verification output missing passed count:\n${VERIFY_OUTPUT}")
endif()

message(STATUS "Verification output:\n${VERIFY_OUTPUT}")
message(STATUS "Verification mode test passed")

# Cleanup
file(REMOVE "${MODEL_FILE}")
message(STATUS "=== Verification Mode Test PASSED ===")
