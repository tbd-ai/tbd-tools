git clone https://github.com/apache/incubator-mxnet.git mxnet --recursive --branch v0.12.0
cp memprofilerv12.patch mxnet/
cd mxnet
git apply memprofilerv12.patch
if [ "$?" -eq 0 ]; then
    echo "[MXNET_TOOLS] Applied profiler patch successfully."
else
    echo "[MXNET_TOOLS] ERROR:Failed to apply memory profiler patch."
    exit -1
fi
echo "done."