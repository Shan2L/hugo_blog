+++
title = 'ubuntu22.04编译安装gcc12.2'
date = 2024-08-18T22:17:29+08:00
draft = false
abstract = "本文记录了如何在ubuntu22.04操作系统上安装gcc12.2的详细步骤。"
categories = ["workflow"]
tags = ["system", "tools", "ubuntu"]
+++


1. 更新apt

```bash
sudo apt-get update
```

2. 安装wget

```bash
sudo apt-get install wget zlib2 make
```

3. 下载相应版本的gcc安装包（把下面的地址换成相应版本的gcc即可）

```bash
wget https://ftp.gnu.org/gnu/gcc/gcc-12.2.0/gcc-12.2.0.tar.gz
```

4. 解压安装包

```bash 
tar -xzvf gcc-12.2.0.tar.gz
```

5. 配置安装路径 sudo vim /etc/profile, 在最后添加

```bash
export PATH="/usr/local/gcc-12.2/bin:$PATH"
```

6. 进入gcc源码目录

```bash
cd gcc-12.2.0/
```

7. 下载需要的配置

```
./contrib/download_prerequisites
```

8. 创建编译目录

```bash
cd  ..

mkdir temp_gcc9.2 && cd temp_gcc9.2
```

这里报错：

```bash
configure: error: no acceptable C compiler found in $PATH
```

原因是没有安装gcc g++， 解决方法：

```bash
apt-get install -y gcc g++
```

再次执行后成功

9. 编译安装

```bash
make && make install 
```

10. 链接

```bash
ln -s /usr/local/gcc-12.2/bin/gcc     /usr/bin/gcc
ln -s /usr/local/gcc-12.2/bin/g++     /usr/bin/g++
```

11. 查看输出结果：

```bash
gcc -v
```

```bash
root@fc4041008a22:/build# gcc -v
Using built-in specs.
COLLECT_GCC=gcc
COLLECT_LTO_WRAPPER=/usr/local/gcc-12.2/libexec/gcc/x86_64-pc-linux-gnu/12.2.0/lto-wrapper
Target: x86_64-pc-linux-gnu
Configured with: ../gcc-12.2.0/configure --prefix=/usr/local/gcc-12.2 --enable-threads=posix --disable-checking --disable-multilib
Thread model: posix
Supported LTO compression algorithms: zlib
gcc version 12.2.0 (GCC)
```

