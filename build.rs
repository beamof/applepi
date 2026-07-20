//! 项目级 build script。
//!
//! 唯一职责：确保链接 C++ 标准库，给 ort（ONNX Runtime）的 C++ 符号兜底。
//!
//! 背景：fastembed → ort → ort-sys 的 build script 在部分 Linux 工具链配置下
//! 不能可靠 emit C++ stdlib link 指令，导致链接 ort 静态库时报
//! `undefined reference to '__cxa_call_terminate'` 等 C++ ABI 符号缺失。
//! 项目层显式补 link 指令。
//!
//! 关键技巧：用 `+` 前缀（`dylib=+stdc++`）把库标记为「整个链接命令的末尾」，
//! 保证它兜底覆盖所有依赖里未解析的 C++ 符号，不受依赖图链接顺序影响。
//! 这是 Cargo 1.65+ 支持的语法。
//!
//! 按 target triple 分发：
//! - Linux / *BSD（gnu ABI）：stdc++
//! - Linux musl：static stdc++（musl 默认静态链接，且需手动接 C++ 库）
//! - macOS：c++
//! - Android：c++_shared
//! - Windows MSVC：不处理（工具链自动）

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();

    if target.contains("msvc") {
        // Windows MSVC：工具链自动链接 C++ 运行时，无需处理。
        return;
    }

    if target.contains("apple") {
        // macOS / iOS：链接 libc++。
        println!("cargo:rustc-link-lib=dylib=+c++");
    } else if target.contains("android") {
        println!("cargo:rustc-link-lib=dylib=+c++_shared");
    } else if target.contains("musl") {
        // musl Linux：默认静态链接；优先尝试静态 libstdc++，失败再回退动态。
        println!("cargo:rustc-link-lib=static=+stdc++");
    } else if target.contains("linux")
        || target.contains("freebsd")
        || target.contains("openbsd")
        || target.contains("netbsd")
    {
        // gnu Linux / *BSD：动态 libstdc++。
        println!("cargo:rustc-link-lib=dylib=+stdc++");
    }
}
