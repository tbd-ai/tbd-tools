git clone https://github.com/apache/incubator-mxnet.git --recursive --branch v0.12.0
cp memprofilerv12.patch incubator-mxnet/
cd incubator-mxnet
git apply memprofilerv12.patch
if [ "$?" -eq 0 ]; then
    echo "Applied profiler patch successfully"
else
    echo "ERROR:Failed to apply memory profiler patch."
fi
echo "done."
