set -e
W=/tmp/krisd/family; cd $W
rm -rf ModelBlaster.new && mkdir ModelBlaster.new && tar xzf mb.tar.gz -C ModelBlaster.new && find ModelBlaster.new -name '._*' -delete
rm -rf ModelBlaster && mv ModelBlaster.new/ModelBlaster ModelBlaster && rmdir ModelBlaster.new
mkdir -p py && ln -sfn $W/ModelBlaster py/modelblaster
[ -d in ] || tar xzf in.tar.gz
echo installed $(date)
