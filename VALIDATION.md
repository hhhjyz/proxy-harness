# v1 验证记录

验证日期：2026-11-25。本次最终测试为 **114 项通过，0 failures，0 errors，0 skipped**。测试时间和源码哈希以 `validation/test-summary.json` 为准；原始测试输出在 `validation/test-suite.log`。

## 验证结论

| 项目 | 结果 | 证据 |
|---|---|---|
| 单元与边界测试 | 见 test-summary；包含真实浏览器用例，最终运行无跳过 | `validation/test-summary.json` |
| 真实 Three.js WebGL 渲染 | 通过 | `validation/demo/iter_00/render_meta.json` |
| 7 个时间点打乱顺序重复渲染 | 像素和 manifest 全部一致 | `validation/demo/iter_00/determinism.json` |
| 程序化 L1 | 通过，0 failures，0 warnings | `validation/demo/iter_00/l1_report.json` |
| MP4 导出 | H.264，640×360，12 fps，36 帧，3 秒 | `validation/demo/export/proxy.mp4` |
| 完整模型生成闭环 | **尚未通过环境联调**：Codex 在初始化时退出 | `validation/live_smoke/iter_00/coder.stderr.log` |
| 独立 VLM 真实评分 | **未取得**：同一 CLI 启动限制；没有伪造分数 | `validation/live_smoke/iter_00/eval.json` |
| 完整 24 条 paired dev / 留出实验 | **未执行** | 暂无通过率或泛化提升结论 |
| MiniMax/H3 推理 | **未执行** | v1 交付 proxy 与 handoff 接口 |

## 演示的准确含义

`validation/demo` 使用项目内手写的球/立方体场景，验证渲染、空间测量、确定性、原始轨迹保存、视频编码和“没有 VLM 就不能接受”的流程。

它的 `hard_pass=true` 表示采样的程序化约束通过；`status=unverified`、`model_mode=demo`、空 VLM 分数是正确结果。演示不进入训练/开发通过率统计，也不能调用正式 `handoff`。

## 模型接入限制

实际检查发现 Codex CLI 已登录，但当前宿主的 CLI 状态目录只读。真实 `codex exec` 返回：

```text
failed to initialize in-process app-server client: Read-only file system (os error 30)
```

启动失败发生在生成前，因此工作目录中残留的初始模板没有被当作模型生成结果。程序向 L1 加入 `output_integrity` 失败，并保存 coder/evaluator 的退出码、stderr、提示和原始事件文件。该运行最终 `status=error`，不能进入技能提炼、准入或视频交接。

项目没有复制认证凭据、改写宿主权限或通过关闭 sandbox 规避限制。请在正常可写、已登录的本地 CLI 环境运行 README 的 `run` 命令；或者配置自己的 coding CLI wrapper 与 HTTP VLM evaluator。此处的适配器代码和本地 `--help` 参数已核对，但成功的真实 CLI continuation 尚未在本宿主得到验证。

## 浏览器来源与实际运行后端

- Python 3.12.14；Playwright 1.62.0；Pillow 13.0.0。
- Three.js 0.180.0 / r180，本地固定依赖。
- 实际浏览器：Chromium 158.0.8127.0。
- 实际 WebGL：ANGLE / Vulkan / SwiftShader。
- 宿主的 `/usr/bin/chromium` 是失效 wrapper，标准 Playwright 下载端点返回 403。因此测试使用已校验包完整性的公开 npm Chromium 分发；二进制未混入交付包。
- 完整来源、integrity、实际浏览器边界测试及受测资产哈希：`docs/browser-validation.json`。

这个替代浏览器解决的是本次宿主的浏览器可用性问题，不是项目的隐式必需依赖。正常环境优先使用 Playwright 安装的浏览器；已有受控浏览器则通过 `--browser` 指定。

## 测试覆盖与没有覆盖的部分

测试覆盖 Spec 规范化、几何假阳性、接触容差、未知碰撞形状、不可用投影、角色/活动区间、低机位与俯仰角、源契约、确定性证据一致性、CLI argv/指定 session、评分拒绝、最佳版本保留、文件保护、train/dev/holdout 隔离、完整配对、候选哈希绑定与复杂度伪改善。

真实浏览器边界测试验证了 OBB/球体缩放、shear、near-plane、behind-camera、修改顶点、未注册 mesh、俯仰角以及顶视图恢复。编排和外循环的模型边界测试使用明确的 mock；它们证明控制逻辑，不证明模型能力或研究指标。

v1 的 L1 是有限时间采样，不是连续碰撞检测。投影面积是包围矩形面积，不是可见像素或遮挡率。复杂人体/手部接触、柔体、骨骼动画、自定义 shader 变形和任意真实物理正确性没有被这些测试证明。相机与身份可读性仍需要真实独立视觉评审。

## 可复现命令

```bash
python -m unittest discover -s tests -v
PROXY_BROWSER=/absolute/path/to/chromium python -m unittest discover -s tests -v
python -m proxy_harness demo --out runs/reproduce-demo --browser /absolute/path/to/chromium --video
python -m proxy_harness run --spec devset/train/train_01_static_pair.spec.json --config configs/codex.json --out runs/real-train-01
```

固定的 6/24/6 测试集是本项目原创工程用例；manifest 规范化哈希为 `2546aa28fbb9296e111bcc4bd8e9058d43b19dbd2ddb4367f7da8acedf78aea9`。不要在模型提炼阶段改用 dev/holdout 输入，也不要仅凭示例视频宣布候选通过。
