#!/bin/bash

base="simoneponcioni@130.92.125.186://home/simoneponcioni/Documents/01_PHD/04_Output-Reports-Presentations-Publications/HFE-RESULTS/repro-results-ubelix/simuation_errors/"
remote="sp20q110@submit01.unibe.ch"

# Read the file paths from the text file
while IFS= read -r file
do
    # Construct the destination directory
    destination="$base$("$file")"

    # Create the destination directory if it doesn't exist
    sudo mkdir -p "$destination"

    # Copy the file
    sshpass -p "$password" sudo scp "$remote:$file" "$destination"
done < files_to_copy.txt
