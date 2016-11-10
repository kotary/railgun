if [ $# -ne 3 ]; then
  echo "Usage: ./deploy.sh [REMOTE_HOST] [REMOTE_DIR] [MAIN_FILE]"
  exit 1;
fi
REMOTE_HOST=$1
REMOTE_DIR=$2
MAIN_FILE=$3

ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR;"
fs=`ls -a1 | grep -v -E "^\.git|^\.+$"`
echo $fs
for f in $fs
do
  scp -P 22 -r ./${f} ${REMOTE_HOST}:${REMOTE_DIR}
done

ssh $REMOTE_HOST "cd $REMOTE_DIR;make clean;make APP=$MAIN_FILE;bin/railgun"
