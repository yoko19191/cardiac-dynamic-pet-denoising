#!/bin/bash

# Get the .pth file with the maximum epoch value by splitting the filenames and sorting numerically
max_epoch_file=$(ls best_model_epoch_*.pth | awk -F_ '{print $4,$0}' | sort -k1,1n | awk '{print $2}' | tail -n 1)

# Display the file with the maximum epoch
echo "The file with the maximum epoch is: $max_epoch_file"

# Ask the user if they want to delete all other .pth files
read -p "Do you want to delete all other .pth files? (yes/no): " choice

# Check if the user entered 'yes'
if [ "$choice" == "yes" ]; then
  # Loop through all .pth files
  for file in best_model_epoch_*.pth
  do
    # If the filename does not match the maximum epoch filename, delete it
    if [ "$file" != "$max_epoch_file" ]; then
      rm -f "$file"
      echo "Deleted $file" # Inform the user of the deletion
    fi
  done
  echo "Done, $max_epoch_file has been retained" # Inform the user that the process is complete
else
  echo "No files have been deleted" # Inform the user that no deletion occurred
fi

