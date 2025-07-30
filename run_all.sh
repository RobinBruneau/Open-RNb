#!/bin/bash

# Define your lists of objects and methods
OBJECTS=("bearPNG" "buddhaPNG" "cowPNG" "pot2PNG" "readingPNG") # Add all your object names here
METHODS=("sdmunips") # "unimsps")       # Add all your method names here
L_ADD_05=("false")
# Path to your run.sh script (now expecting two arguments)
RUN_SCRIPT="./train.sh"
TAG=""
count=1



for i in $(seq $count); do
    # Loop through each object
    for ADD_05 in "${L_ADD_05[@]}"; do

        if [ "$ADD_05" == "true" ]; then
            TAG="_05_crop"
        else 
            TAG="_00_crop"
        fi

        for object in "${OBJECTS[@]}"; do
            # Loop through each method for the current object
            for method in "${METHODS[@]}"; do
                SCENE="dlmv_data/${object}/${method}"
                
                # Determine NO_ALBEDO value based on the method
                NO_ALBEDO_VALUE=""
                if [ "$method" == "sdmunips" ]; then
                    NO_ALBEDO_VALUE="true"
                elif [ "$method" == "unimsps" ]; then
                    NO_ALBEDO_VALUE="true"
                else
                    echo "Warning: Unknown method '$method'. Defaulting NO_ALBEDO to 'false'."
                    NO_ALBEDO_VALUE="false" # Default value if method is not recognized
                fi

                echo "Launching $RUN_SCRIPT with scene: $SCENE and NO_ALBEDO: $NO_ALBEDO_VALUE"
                # Execute the train.sh script with the scene path and the determined NO_ALBEDO value
                bash "$RUN_SCRIPT" "$SCENE" "$NO_ALBEDO_VALUE" "$ADD_05" "$TAG"
                echo "-------------------------------------"
            done
        done
    done
done
echo "All scene combinations processed!"
