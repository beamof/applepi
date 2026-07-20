//! 项目级 build script。
//!
//! 唯一职责：把 C++ 标准库追加到**最终 bin 链接命令行的末尾**，给 ort
//! （ONNX Runtime）静态库里的 C++ 符号（`__cxa_call_terminate` 等）兜底。
//!
//! ## 为什么用 `rustc-link-arg-bins` 而不是 `rustc-link-lib`
//!
//! 之前版本用 `cargo:rustc-link-lib=dylib=stdc++`，它会被放到链接命令行
//! **开头**。但 GNU ld 是单趟解析：当命令行后面的 ort rlib 引用了
//! `__cxa_call_terminate` 时，前面已扫过的 libstdc++ 不会被重新查找，导致
//! `undefined reference to '__cxa_call_terminate'`。
//!
//! `cargo:rustc-link-arg-bins=<arg>` 会把 `<arg>` 原样追加到**每个 bin
//! crate 的链接命令行末尾**（在所有依赖 rlib 之后），符号解析顺序正确。
//!
//! ## 为什么不再用 `.cargo/config.toml` 的 `rustflags`
//!
//! `rustflags = ["-C", "link-arg=-lstdc++"]` 虽然也是直接加 link-arg，但
//! 它作用在**所有 crate**（包括 ort-sys 自己的 build script 链接），而且
//! 同样会面临位置不确定的问题。`rustc-link-arg-bins` 精准作用于最终 bin，
//! 是最稳的层次。
//!
//! 按 target triple 分发：Linux / *BSD → stdc++，macOS → c++，Android →
//! c++_shared，Windows MSVC 不处理（工具链自动链接）。

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();

    if target.contains("msvc") {
        // Windows MSVC：工具链自动链接 C++ 运行时，无需处理。
        return;
    }

    let arg = if target.contains("apple") {
        "-lc++"
    } else if target.contains("android") {
        "-lc++_shared"
    } else if target.contains("musl") {
        // musl 默认静态链接：先尝试静态 libstdc++，链接器找不到会回退动态。
        "-Wl,-Bstatic -lstdc++ -Wl,-Bdynamic"
    } else if target.contains("linux")
        || target.contains("freebsd")
        || target.contains("openbsd")
        || target.contains("netbsd")
    {
        // gnu Linux / *BSD：动态 libstdc++。
        "-lstdc++"
    } else {
        return;
    };

    // 追加到每个 bin crate 的链接命令行末尾（在所有 rlib 之后）。
    // bin 名固定为 `bot`、`cli`（见 src/bin/）。也可以不指定名字，对所有 bin 生效。
    println!("cargo:rustc-link-arg-bins={arg}");
    println!("cargo:rerun-if-changed=build.rs");
}
