#!/usr/bin/sh
# This script will purge the files older than given days from a given location with a filename satisfying a regex.
# Deployment steps --> copy this file at /usr/share/
# change the permission  ---> chmod 755 purgeOldFiles.sh
#  setup as crontab with following syntax
# */1 * * * * sh /usr/share/purgeOldFiles.sh >> /tmp/purgeOldFiles.log 2>&1   (this script will run every minute)
usage ()
{
  echo 'Usage : Script  <path> <file name regex> <duration in days>'
  echo 'For example below command will delete all log files inside /var/log/ location (recursively)which are older than 9 days '
  echo 'sh purgeOldFiles.sh  /var/log *.log 9'

  exit
}
if [ "$#" -ne 3 ]
    then
      usage
fi

FILES_TO_BE_PURGED_CMD="find $1/ -name \"$2\" -type f -mtime +"$3";"
echo "command generated for listing files eligible for purging: \n $FILES_TO_BE_PURGED_CMD"
FILES_TO_BE_PURGED=$(eval "$FILES_TO_BE_PURGED_CMD")
echo "Below files are eligible to be purged..."

echo $FILES_TO_BE_PURGED

echo "\n\n"

echo "Starting purging files at "$(date '+%Y%m%d%H%M%S')
PURGE_CMD="find $1/ -name \"$2\" -type f -mtime +"$3" -exec rm -f {} \;"
echo "command generated for purging older file is :\n $PURGE_CMD"
echo "executing the command now..."
eval "$PURGE_CMD"
echo "Finished purging files at "$(date '+%Y%m%d%H%M%S')