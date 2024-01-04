#!/bin/bash

# Wrapper to run relion_external_reconstruct and sidesplitter
# Version 1.3
# Author: Colin M. Palmer & Takanori Nakane

# Usage:
#     sidesplitter_wrapper.sh path/to/relion_external_reconstruct_star_file.star
#
# To use from RELION 3.1, set the RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE environment variable to point to this script,
# set SIDESPLITTER to point to the sidesplitter binary (or make sure sidesplitter can be found via PATH), and run
# relion_refine with the --external_reconstruct argument. For example:
#
#     export RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE=/path/to/sidesplitter_wrapper.sh
#     export SIDESPLITTER=/path/to/sidesplitter
#
# then run RELION auto-refine from the GUI and put "--external_reconstruct" in the additional arguments box. To run on
# a cluster, depending on your configuration you might need to put the environment variable definitions into your
# submission script.

# Troubleshooting
#
# If you have problems running SIDESPLITTER using this script, the first thing to check is that external reconstruction
# from RELION is working correctly. Try running a normal refinement job, using the "--external_reconstruct" argument
# but without setting the RELION_EXTERNAL_RECONSTRUCT_EXECUTABLE environment variable. If this fails, the problem is
# likely to be with your RELION installation - perhaps it is the wrong version, or different installations are
# conflicting with each other. If normal external reconstruction is successful, the problem is likely to be with the
# SIDESPLITTER installation, or a bug in this script.

# How this script works:
#
# If the target file name contains "_half", this script assumes two copies of itself will be running (for the two half
# data sets). The two copies will coordinate with each other by creating, checking for and deleting a directory called
# "sidesplitter_running". Both scripts will run relion_external_reconstruct for their given half data set. The first
# script will wait for both reconstructions to finish, then call SIDESPLITTER to process both half maps. The second
# script will wait for the first to finish running SIDESPLITTER and then exit (because if either of the scripts exits
# before the processing is finished, RELION moves on and tries to continue its own processing before the filtered
# volumes are ready).
#
# If the target file name does not contain "_half", this script assumes there is only a single copy of itself running.
# In this case it calls relion_external_reconstruct, waits for the reconstruction to finish and then exits.
# This handles the final iteration when the two half sets are combined, at which point RELION calls the external
# reconstruction program just once to reconstruct the final combined volume.
#
# Note that this script is not particularly robust. If one of the commands fails, it's possible you might need to
# manually tidy up the job directory and remove the "sidesplitter_running" directory to avoid problems in the next run.


#### Configuration

# SIDESPLITTER command ("sidesplitter" unless defined in the SIDESPLITTER environment variable)
sidesplitter_default=sidesplitter
sidesplitter=${SIDESPLITTER:-${sidesplitter_default}}

# Change to true to activate debug output (to check for problems with process coordination)
debug=false

#### Main script

# Convenience function to print and run a command
echo_and_run() {
  echo "$$ $(date) Running $@"
  eval "$@"
}

# Expect this to be called with one argument, pointing at a STAR file for relion_external_reconstruct
base_name=$(basename -- "$1" ".star")
job_dir=$(dirname -- "$1")
base_path="${job_dir}/${base_name}"
base_dir=`dirname $base_path`
body_id=$(echo ${base_name} | sed -n 's,^.*_body\([0-9]*\)_external_reconstruct.*$,\1,p')

# Find the mask
# TODO: Do this only once!
mask=""
if [ -z "$body_id" ]; then # multibody
  mask=$(awk '/fn_mask/ { gsub(/^"|"$/, "", $2); print $2 }' ${base_dir}/job.star)
else
  fnbody=$(awk '/fn_bodies/ { gsub(/^"|"$/, "", $2); print $2 }' ${base_dir}/job.star)
  origmask=$(relion_star_printtable $fnbody data_ rlnBodyMaskName | sed "${body_id}q;d")
  mask=${base_dir}/centered_mask_body${body_id}.mrc
  relion_image_handler --i $origmask --o $mask `relion_image_handler --i $origmask --com | awk '{print "--shift_x "(-$10)" --shift_y "(-$12)" --shift_z "(-$14)}'`
fi

# Because only one of the two performs the actual work, use twice the number of threads.
nthreads=$(awk '/nr_threads/ { gsub(/^"|"$/, "", $2); print 2*$2 }' ${base_path%/*}/job.star)

$debug && echo NTHREADS=$nthreads
$debug && echo BODY_ID=$body_id
$debug && echo MASK=$mask

# Create a name to use for Sidesplitter output by removing "_half1" or "_half2" and "_external_reconstruct" from base_name
name_without_half="${base_name/_half[12]/}"
sidesplitter_base="${job_dir}/${name_without_half/_external_reconstruct/}_sidesplitter"

running_indicator_dir="${sidesplitter_base}_running"

$debug && echo "$$ Checking for existence of $running_indicator_dir ..."

if mkdir "$running_indicator_dir" 2> /dev/null; then
  first=true
  $debug && echo "$$ Created $running_indicator_dir"
else
  first=false
  $debug && echo "$$ $running_indicator_dir already exists"
fi

echo_and_run "relion_external_reconstruct \"$1\" > \"${base_path}.out\" 2> \"${base_path}.err\""

if [[ $base_name != *"_half"* ]]; then
  if $first; then
    $debug && echo "$$ $(date) Only a single reconstruction, removing $running_indicator_dir and exiting."
    rmdir "$running_indicator_dir"
    exit 0
  else
    $debug && echo "$$ $(date) Error! Found pre-existing $running_indicator_dir for single reconstruction job."
    exit 1
  fi
fi

$debug && echo "$$ Moving output file ${base_path}.mrc to ${base_path}_orig.mrc"
mv "${base_path}.mrc" "${base_path}_orig.mrc"

#### START OF FSC MODIFICATION SEGMENT ####

# Inflate FSC by FSC_SS = (sqrt(2 (FSC + FSC^2)) + FSC) / (2 + FSC)
# Comment this out if FSC modification is not desirable - e.g. for particularly poorly stable refinements

outstar=`awk '/_rlnExtReconsResultStarfile/ {print $2}' $1`
echo "Writing adjusted FSC curve to $outstar"

echo "data_

loop_
_rlnSpectralIndex #1
_rlnGoldStandardFsc #2
_rlnFourierShellCorrelation #3
" > $outstar

relion_star_printtable $1 data_external_reconstruct_tau2 rlnSpectralIndex rlnGoldStandardFsc | awk '{$3 = (sqrt(2 * ($2 + $2 * $2)) + $2) / (2 + $2); print $0}' >> $outstar

####  END OF FSC MODIFICATION SEGMENT  ####

if $first; then
  $debug && echo "$$ $(date) First reconstruct job finished; waiting for $running_indicator_dir to disappear"
  while [[ -d $running_indicator_dir ]]; do
    $debug && echo "$$ $(date) $running_indicator_dir still exists; waiting..."
    sleep 5
  done

  $debug && echo "$$ $(date) $running_indicator_dir has disappeared. Moving on to SIDESPLITTER step."

  # Prepare the SIDESPLITTER command
  half1_basename=${base_path/half2/half1}
  half2_basename=${base_path/half1/half2}
  sidesplitter_command="OMP_NUM_THREADS=$nthreads $sidesplitter --v1 \"${half1_basename}_orig.mrc\" --v2 \"${half2_basename}_orig.mrc\" --rotfl"
  if [[ -z "$mask" ]]; then
    echo "Warning: no mask found! SIDESPLITTER will give better results if you use a mask."
  else
    sidesplitter_command="${sidesplitter_command} --mask \"${mask}\""
  fi
  sidesplitter_command="${sidesplitter_command} > \"${sidesplitter_base}.out\""

  # Run SIDESPLITTER
  echo_and_run "$sidesplitter_command"

  # Move output files to where RELION expects them
  $debug && echo "$$ Moving sidesplitter output ${half1_basename}_orig_sidesplitter.mrc to ${half1_basename}.mrc"
  mv ${half1_basename}_orig_sidesplitter.mrc ${half1_basename}.mrc

  $debug && echo "$$ Moving sidesplitter output ${half2_basename}_orig_sidesplitter.mrc to ${half2_basename}.mrc"
  mv ${half2_basename}_orig_sidesplitter.mrc ${half2_basename}.mrc

  $debug && echo "$$ $(date) Finished sidesplitter. Recreating $running_indicator_dir to indicate job finished."
  mkdir "$running_indicator_dir"

else
  $debug && echo "$$ $(date) Second reconstruct job finished; removing $running_indicator_dir"
  rmdir "$running_indicator_dir"
  $debug && echo "$$ $(date) Waiting for $running_indicator_dir to reappear"
  while [[ ! -d $running_indicator_dir ]]; do
    $debug && echo "$$ $(date) $running_indicator_dir does not exist; waiting..."
    sleep 60
  done
  $debug && echo "$$ $(date) $running_indicator_dir has reappeared. Removing it and exiting"
  rmdir "$running_indicator_dir"
fi
