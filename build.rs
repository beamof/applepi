//! 项目级 build script。
//!
//! 唯一职责：确保链接 C++ 标准库，给 ort（ONNX Runtime）的 C++ 符号兜底。
//!
//! 背景：fastembed → ort → ort-sys 的 build script 在部分 Linux 工具链配置下
//! 不能可靠 emit C++ stdlib link 指令，导致链接 ort 静态库时报
//! `undefined reference to '__cxa_call_terminate'` 等 C++ ABI 符号缺失。
//!
//! ort-sys 声明 `links = "onnxruntime"`，理论上会自己发 stdc++ 指令，但实际
//! 在某些路径（download-binaries + 特定 target）下不会触发。项目层兜底补一条。
//!
//! 按 target triple 分发：Linux / *BSD → stdc++，macOS → c++，Android → c++_shared，
//! Windows MSVC 不处理（工具链自动链接）。

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();

    if target.contains("msvc") {
        // Windows MSVC：工具链自动链接 C++ 运行时，无需处理。
        return;
    }

    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("android") {
        println!("cargo:rustc-link-lib=c++_shared");
    } else if target.contains("musl") {
        // musl 默认静态链接；尝试静态 libstdc++，再由链接器兜底回退动态。
        println!("cargo:rustc-link-lib=static=stdc++");
    } else if target.contains("linux")
        || target.contains("freebsd")
        || target.contains("openbsd")
        || target.contains("netbsd")
    {
        // gnu Linux / *BSD：动态 libstdc++。
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}
