#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source installed ROS distro setup
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO}/setup.sh"

# Activate our workspace
source /angel_workspace/workspace_setenv.sh

# If CYCLONE_DDS_INTERFACE is defined to a value, then template
if [[ -n "$CYCLONE_DDS_INTERFACE" ]]
then
  envsubst <"${SCRIPT_DIR}/cyclonedds_profile.xml.tmpl" >"${SCRIPT_DIR}/cyclonedds_profile.xml"
  export CYCLONEDDS_URI=file://${SCRIPT_DIR}/cyclonedds_profile.xml
fi

exec "$@"