#!/bin/bash
set -e

cd /hy-tmp/Mapdiff-repo
# 压缩包名称
file="outputs-$(TZ=Asia/Shanghai date "+%Y%m%d-%H%M").zip"
# 把 output 目录做成 zip 压缩包
zip -q -r "${file}" outputs

# 上传压缩包到网盘 
gpushare-cli ali up /hy-tmp/Mapdiff-repo/"${file}" /mapdiff/

# 检查传输结果,成功则关机，失败则挂起
if gpushare-cli ali ls /mapdiff/ | grep -Fq "${file}"; then
       rm -f "${file}"
	shutdown	
else
	echo "Save failed, remote service is hold"
	exit 1
fi
