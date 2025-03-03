#!/bin/bash

# Maximum number of retry attempts
MAX_RETRIES=200
# Delay between retries (in seconds)
RETRY_DELAY=30
# Installation command
INSTALL_CMD="pip install git+https://github.com/NVIDIA/NeMo.git@r1.23.0#egg=nemo_toolkit[asr]"

# Function to install with retries
install_with_retry() {
    local retries=0
    local success=false

    echo "Starting NeMo Toolkit installation with retry mechanism..."
    echo "Maximum retries: $MAX_RETRIES"
    echo "Delay between retries: $RETRY_DELAY seconds"
    
    # Configure git settings for better network tolerance
    git config --global http.lowSpeedLimit 1000
    git config --global http.lowSpeedTime 300
    
    while [ $retries -lt $MAX_RETRIES ] && [ "$success" = false ]; do
        echo "------------------------------------------------------------"
        echo "Attempt $(($retries + 1)) of $MAX_RETRIES"
        echo "Running: $INSTALL_CMD"
        echo "------------------------------------------------------------"
        
        if $INSTALL_CMD; then
            echo "Installation successful!"
            success=true
        else
            retries=$((retries + 1))
            if [ $retries -lt $MAX_RETRIES ]; then
                echo "Installation failed. Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
            else
                echo "Maximum retry attempts reached. Installation failed."
            fi
        fi
    done
    
    if [ "$success" = true ]; then
        return 0
    else
        return 1
    fi
}

# Execute the installation function
install_with_retry
exit_code=$?

# Final output
if [ $exit_code -eq 0 ]; then
    echo "------------------------------------------------------------"
    echo "NeMo Toolkit installation completed successfully!"
    echo "------------------------------------------------------------"
else
    echo "------------------------------------------------------------"
    echo "NeMo Toolkit installation failed after $MAX_RETRIES attempts."
    echo "Please check your network connection and try again later."
    echo "------------------------------------------------------------"
fi

exit $exit_code