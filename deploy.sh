REMOTE_HOST=$1
REMOTE_DIR=$2

ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR;"
fs=`ls -a1 | grep -v -E "^\.git|^\.+$"`
echo $fs
for f in $fs
do
  scp -P 22 -r ./${f} ${REMOTE_HOST}:${REMOTE_DIR}
done

ssh $REMOTE_HOST "cd $REMOTE_DIR;make clean;make;bin/railgun"
