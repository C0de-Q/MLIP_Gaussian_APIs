# 理想的项目 Git 工作流

不存在唯一「官方标准」的工作流。Git 官方文档给出的是一组规则和可拼接的模式（`gitworkflows` 手册页），
真正被广泛推荐的组合是：**主干常绿 + 短生命周期分支 + Pull Request 评审 + CI 校验 + 用标签做发布**。
分支模型怎么选，取决于你是一次性交付的 Web/服务型项目，还是需要长期维护多个已发布版本的软件。

## 目录

- 工作流由哪几件事构成
- Git 官方给出的一般性规则
- 仓库之间的三种协作模式
- 四种主流分支模型对比
- 提交信息与版本号
- 合并方式：merge / squash / rebase
- 常见坑
- 建议与选型
- 参考资料
- 未核实之处

## 工作流由哪几件事构成

「工作流」不是一个开关，而是四个可独立决定的维度：

| 维度 | 需要决定的事 |
| --- | --- |
| 分支模型 | 有几条长期分支？功能在什么地方开发、回到哪里？ |
| 集成方式 | 用 merge commit、squash 还是 rebase 把改动并回主干？ |
| 评审与校验 | 是否强制 PR 评审、是否要求 CI 通过后才能合并？ |
| 发布节奏 | 持续交付，还是按版本发布并需要维护旧版本？ |

## Git 官方给出的一般性规则

Git 自己的 `gitworkflows` 手册页（维护 Git 项目本身所用的模型）给出了两条明确规则，它们是多数工作流的共同底层逻辑：

- **Merge upwards（向上合并）**：把修复提交到「最老的、需要这个修复的受支持分支」，然后定期把集成分支向上逐级合并。
- **Topic branches（主题分支）**：每个功能/修复都开一条侧分支，并且**从你最终想合并进的那条最老的集成分支上分叉**。

它还给出了 Git 项目自身的「集成分支分级」与 **Graduation（毕业）** 概念：改动先进入较不稳定的分支，
稳定后再「毕业」到更稳定的分支。

| 分支 | 用途 |
| --- | --- |
| `seen` | 维护者已看到、但还不够成熟的补丁 |
| `next` | 正在为进入 `master` 做稳定性测试的主题 |
| `master` | 将要进入下一次发布的提交 |
| `maint` | 将进入下一次维护版（旧稳定版更新）的提交 |

同一页面还有一条与 rebase 直接相关的警告：**已经被合并到别处的主题分支，不要再 rebase。**

## 仓库之间的三种协作模式

Pro Git 的「分布式工作流」把仓库级协作分成三种，选择它决定了你是否需要 fork、谁有推送权限：

| 模式 | 形态 | 适用 |
| --- | --- | --- |
| 集中式 | 单一中心仓库，所有人有推送权限，后推的人必须先合并 | 小团队、从 SVN 迁移的团队 |
| 集成管理者 | 每个开发者有自己的公开仓库，向维护者发「请拉取我的改动」请求 | GitHub/GitLab 上的 fork + PR 模型 |
| 独裁者与副官 | 多个集成管理者（副官）负责不同子系统，最终由一人汇总 | 像 Linux 内核这样的超大型项目 |

GitHub 上的 fork + Pull Request 本质上就是「集成管理者」模式的产品化实现。

## 四种主流分支模型对比

| 维度 | Git Flow | GitHub Flow | GitLab Flow | Trunk-Based |
| --- | --- | --- | --- | --- |
| 长期分支 | `master` + `develop` 两条 | 只有默认分支 | 主干加按环境/版本派生的分支（未能核实） | 只有主干（`main`/trunk） |
| 主要分支类型 | feature / release / hotfix | 短名称功能分支 | 功能分支 + 环境分支（未能核实） | 短生命周期功能分支，或直接提交 |
| 发布方式 | 从 `release` 分支发版、打标签 | 合并到默认分支即可部署 | 按环境逐级合并（未能核实） | 从主干发布，或临时切 release 分支 |
| 适合场景 | 需要同时维护多个已发布版本的软件 | 持续交付的 Web/服务型项目 | 有 staging/production 多环境、自建 GitLab | 追求高频集成、CI/CD 成熟的团队 |
| 主要代价 | 分支多、生命周期长，容易堆积合并冲突 | 对测试与 CI 依赖强 | —— | 要求提交粒度小、有构建服务器兜底 |

需要特别说明的是：**Git Flow 的作者本人在 2020 年给原文加了一段「Note of reflection」**，
指出这个模型诞生于 2010 年，如今最流行的软件形态变成了持续交付的 Web 应用，
如果你在做持续交付，他建议改用更简单的流程（例如 GitHub Flow），不要硬套 git-flow；
而如果你在做「明确版本化」、需要支持多个线上版本的软件，git-flow 依然合适。他同时强调：**不存在万灵药，要结合自己的语境**。

Trunk-Based Development 的定义（来自其官方站点）是：开发者只在一条被称为 trunk 的分支上协作，
并通过若干成文的技巧**抵抗创建其他长期分支的诱惑**，从而避免 merge 地狱、不破坏构建。
它强调为了让持续集成成立，团队成员应至少每 24 小时向主干提交一次；
小团队可以直接提交到主干，其他情况用「短生命周期功能分支 + PR」做评审和 CI，
但这种分支只用于评审/构建检查，**不用于产出发布物**。发布节奏需要时再按需切出 release 分支并在发布后删除。

## 提交信息与版本号

两套被广泛采用的约定，一起用效果最好：

- **Conventional Commits 1.0.0**：提交信息结构为 `<type>[optional scope]: <description>`，可带 body 与 footer。
  `fix:` 对应补丁，`feat:` 对应新功能，footer 里的 `BREAKING CHANGE:`（或在 type/scope 后加 `!`）表示破坏性变更。
- **Semantic Versioning**：`MAJOR.MINOR.PATCH` —— 不兼容的 API 变更升 MAJOR，向后兼容地新增功能升 MINOR，向后兼容地修 bug 升 PATCH。

两者是互补的：Conventional Commits 让「哪些提交属于哪一类变更」可被机器解析，从而自动决定版本号怎么升，
这正是 Conventional Commits 规范里明确写出的设计意图。

```
feat(mlip_server): support DPA4 checkpoints

fix: release the model when a Gaussian job exits early

feat(api)!: drop the legacy gau_scripts entry point
```

## 合并方式：merge / squash / rebase

GitHub 官方文档对三种合并方式的取舍给出了明确说明：

- **Merge commit（默认）**：保留完整历史与合并点；如果要求线性历史，可以在受保护分支上启用 "require linear history"。
- **Squash and merge**：历史干净，但代价是——丢失原始提交的时间与作者信息；
  如果 squash 后继续在同一条长期分支上工作，新 PR 里会重复出现已被 squash 的提交，并可能反复冲突；
  原始 SHA 消失后，`git rerere` 之类依赖哈希的命令效果会变差。
- **Rebase and merge**：需要贡献者在命令行 rebase、解决冲突并 force push 到自己的分支，因此必须谨慎，避免覆盖他人已有的工作。

## 常见坑

- **长期分支是合并冲突的温床**：官方 `gitworkflows` 明确指出，把提交直接堆在集成分支上会导致坏提交难以回退、
  并行工作互相搅乱；这正是「每个主题开侧分支 + 频繁向上合并」的动机。
- **rebase 已推送的历史**：Pro Git 的黄金法则是「不要 rebase 已经存在于你的仓库之外、别人可能已在其上工作的提交」。
  `gitworkflows` 的对应说法是：已被合并到别处的主题分支不要再 rebase。
- **squash 不是免费的**：见上一节的三个代价，尤其是长期分支上的重复冲突。
- **默认分支不做保护**：没有评审要求、没有 CI 门禁的 `main`，会让上面所有规则形同虚设。
- **`master` 与 `main` 并存**：两个名字指向不同历史时才会出现「不相关历史」，需要用 `--allow-unrelated-histories` 才能合并。
  一旦两段历史接上了（一方成为另一方的祖先），后续合并就只是普通的快进，不再需要这个参数。让两套命名长期并存只会持续制造混乱。
- **环境相关的坑**：本机 `/etc/ssh/ssh_config` 里 `Host *` 段写过 `Port 2222`，会让 `git pull/push` 连 github.com 时走错端口而超时；
  这是环境配置问题，不是 Git 的问题，可在 `~/.ssh/config` 里用 `Host github.com` + `Port 22` 覆盖。
- **把大文件交给 Git**：模型权重、轨迹、大体积数据文件放进仓库会让 clone 越来越慢。（本条为一般工程经验，不在下列引用来源中，此处不做展开。）

## 建议与选型

如果只记一条结论：**除非你需要同时维护多个已发布版本，否则不要用多长期分支的复杂模型。**

选型可以按下面的判断走：

- 持续交付的服务/工具、团队小：**GitHub Flow**（等价于轻量 Trunk-Based）。
- 有 staging/production 多环境或自建 GitLab：GitLab Flow 思路（其官方页面本次未能打开，见文末）。
- 需要明确版本发布并维护旧版本（例如要发布给他人引用的科学计算库）：**Git Flow**，或「主干 + 按需 release 分支」的简化版。
- 追求高频集成、已有 CI/CD：**Trunk-Based**，配合 feature flag 与 branch by abstraction。

对一个人或几个人的科研代码项目，推荐这套最小可用组合：

1. 只保留一条长期分支 `main`，任何时刻都处于可运行状态。
2. 每个改动开一条短分支，命名带类型前缀：`feat/dpa4-support`、`fix/server-exit-hang`。
3. 合并前必须过 CI（哪怕只有 `pytest` 和一条冒烟脚本），合并用 squash 或 rebase 保持线性历史。
4. 提交信息遵循 Conventional Commits，发布时打标签并给出语义化版本：`git tag -a v0.1.0 -m "first public release"`。
5. 在仓库设置里保护默认分支：禁止直接 push，要求 PR 与 CI 通过。

落到本仓库当前的状态，收敛步骤是：

```bash
# 本地 master 改名为 main（历史已经是线性的，无需再合并）
git branch -M main
git push -u origin main
# 远端 main 追上后，删掉旧的 master（可选）
git push origin --delete master
# 之后每次改动都按「短分支 -> PR -> squash 合并」走
git switch -c feat/xxx
```

并在 GitHub 的 Settings → Branches 把默认分支设为 `main`，其余按上面的第 5 条配置保护规则。

## 参考资料

- Git 官方手册 `gitworkflows`：<https://git-scm.com/docs/gitworkflows>
- Pro Git · Distributed Workflows（集中式 / 集成管理者 / 独裁者与副官）：<https://git-scm.com/book/en/v2/Distributed-Git-Distributed-Workflows>
- Pro Git · Branching Workflows（长期分支与主题分支）：<https://git-scm.com/book/en/v2/Git-Branching-Branching-Workflows>
- Pro Git · Rebasing（"Do not rebase commits that exist outside your repository…"）：<https://git-scm.com/book/en/v2/Git-Branching-Rebasing>
- GitHub 官方文档 · GitHub flow：<https://docs.github.com/en/get-started/using-github/github-flow>
- GitHub 官方文档 · About merge methods on GitHub：<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/about-merge-methods-on-github>
- Vincent Driessen · A successful Git branching model（含 2020 年的 Note of reflection）：<https://nvie.com/posts/a-successful-git-branching-model/>
- Trunk-Based Development 官方站点：<https://trunkbaseddevelopment.com/>
- Conventional Commits 1.0.0：<https://www.conventionalcommits.org/en/v1.0.0/>
- Semantic Versioning 2.0.0：<https://semver.org/>
- Atlassian · Comparing Git workflows（二次来源，用于集中式/功能分支/forking 的背景描述）：<https://www.atlassian.com/git/tutorials/comparing-workflows>

## 未核实之处

- **GitLab Flow**：官方说明页 <https://docs.gitlab.com/topics/gitlab_flow/> 抓取时被 Cloudflare 拦截，
  返回 403（页面标题为 "Just a moment..."）；`about.gitlab.com` 对应页面无法建立连接。
  因此上表中 GitLab Flow 的行是留白/标注「未能核实」，没有依据官方文本填写内容，请以 GitLab 官方文档为准。
- GitLab 官方仓库的文档源文件路径（`doc/topics/gitlab_flow.md`）已不存在，API 搜索需要认证，均未能取到替代来源。
- 上文「大文件不要进 Git」一条属于一般工程经验，未引用上述任何来源，故未展开为具体方案。
