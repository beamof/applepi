//! 项目级 build script。
//!
//! 唯一职责：确保 Linux 上链接 C++ 标准库（libstdc++）。
//!
//! 背景：fastembed → ort → ort-sys 的 build script 在某些 Linux 工具链配置下
//! 不会可靠地 emit `cargo:rustc-link-lib=stdc++`，导致链接 ort 静态库时报
//! `undefined reference to '__cxa_call_terminate'` 等 C++ ABI 符号缺失。
//! 这里在项目层兜底补一条 link 指令，跨发行版/工具链稳定。
//!
//! 只针对非 Apple/非 Android 的 Unix（即 Linux 等）发指令；
//! Windows (MSVC) 和 macOS 由各自的工具链/ort-sys 处理。

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple") {
        // macOS: 链接 libc++（ort-sys 自己会发，但兜底无妨）
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("android") {
        println!("cargo:rustc-link-lib=c++_shared");
    } else if target.contains("msvc") {
        // Windows MSVC: 由工具链自动链接，无需处理
    } else if target.contains("linux") || target.contains("freebsd") || target.contains("openbsd") {
        // Linux 等 Unix：链接 libstdc++
        println!("cargo:rustc-link-lib=stdc++");
    }
}
