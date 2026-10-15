# Publishing to Bilibili and 小红书 (only when the user asks)

The video is delivered to `~/Desktop` first. Publish only when the user asks ("投稿吧", "小红书和 B 站都投一下").
Both sites are driven in the in-app browser pane, where the user is already logged in.

## Hard rules (user-confirmed)

- **小红书: every note 标记 SGLang.** Tag the official SGLang account through 添加组件 → 标记地点或标记朋友, so
  the note appears on SGLang's 标记墙 (see §小红书). Never publish a 小红书 note without the tag.
- **小红书: never @SGLang in the body.** An @ is not a tag. Each @ only sends the account a notification, and it
  does not put the note on the 标记墙.
- Creation statement is always "AI-generated" on both sites: B站 创作声明 = 含AI生成内容, 小红书 内容类型声明 =
  笔记含AI合成内容. The description says 「中文讲解 + 中英双语字幕；配音为 AI 语音合成。」.
- Never mention PR status (OPEN / in review) in titles or descriptions.

## Who does what

The agent cannot put a local file into a native `<input type=file>`: `form_input` fails, and the Chrome
extension caps uploads at 10 MB. It also must not script the OS file dialog. So the agent clicks the upload
button, the user picks the file in the native dialog and replies 「好了」, and the agent does everything else.

Copy the deliverables into one folder so the user always picks from the same place:
`~/Desktop/小红书上传/NN_<slug>.mp4`, `NN_<slug>_封面.png`, `NN_<slug>_封面_B站.png` (NN = publish order).

## Bilibili (`member.bilibili.com/platform/upload/video/frame`)

- **Viewport:** call `resize_window 1280x1000` first. The pane is narrow and the app resets the size every turn.
- **标题 (≤ 80 chars):**
  - PR: 「【SGLang PR 讲解】<一句问题> | PR #N」
  - Article: 「<标题>：<一句问题> | SGLang 技术解读（中英字幕）」
- **创作声明:** pick 含AI生成内容, then Escape and click a blank spot to close the dropdown.
- **分区:** pick 人工智能. Bilibili often preselects vlog or 科技数码. The 人工智能 option sits off-screen inside the
  list, so `scrollIntoView` the `p.item-cont-main` whose text is 人工智能 before clicking it.
- **Tags:** remove the auto tags (原创, 影视剪辑, 学习, 英语, 教育…) from right to left. Then add about 8 tags, each as
  type → wait 1 s → Enter → wait 1.5 s. Batching faster drops every other tag.
- **简介 (quill `.ql-editor`), in this order:**
  - one summary sentence;
  - 内容：… ;
  - 结果（取自 PR / 原文）：… ;
  - the AI-voice line;
  - full links (原文 / Cookbook / PR).
- **封面:**
  - Open 封面设置 → 上传封面, and the user picks `NN_<slug>_封面_B站.png`.
  - Verify the two `x/vu/web/cover/up` responses, one 1440×1080 and one 1440×810, by opening both URLs.
- **Submit:** 立即投稿 → 「稿件投递成功」.
- **BV id:** read it from `member.bilibili.com/x/web/archives?status=is_pubing,pubed,not_pubed&pn=1&ps=3`.

## 小红书 (`creator.xiaohongshu.com/publish/publish?source=official&target=video`)

- **Viewport and focus:**
  - Call `resize_window 1440x1000`. The cover editor needs the width.
  - Focus fields through JS (`input[placeholder="填写标题会有更多赞哦"]`, `.tiptap.ProseMirror`), not by
    coordinates. A pane resize mid-batch makes coordinate clicks miss.
- **标题:** ≤ 20 characters, with ASCII counted as half.
- **正文:**
  - Same content as the Bilibili description.
  - No 「#」 directly before digits, because it opens the topic popup. Write 「PR 44051」.
  - Write links as plain text, e.g. 「PR：sgl-project/sglang PR 44051」, 「原文：SGLang Blog · <slug>」.
- **话题 (5–6 topics, at the end of the body):**
  - For each: type `#词` → wait 1.5 s → check `.item.is-selected` is the exact topic → Enter. The topic becomes
    `a.tiptap-topic`.
  - Always include #人工智能 #大模型 #SGLang.
- **内容类型声明:** 添加内容类型声明 → 笔记含AI合成内容.
- **封面:**
  - 小红书 crops to 4:3 and uses the video's first frame (our cover frame) as a 1440×1080 「已上传封面」.
  - Open 编辑封面 and compare it with `NN_<slug>_封面_B站.png`.
  - If it differs, click 清除已上传封面 → 上传, and the user picks the B站 cover. The editor sometimes opens by
    itself after the upload; typing into the body still works behind it.
- **标记 SGLang (required), in 添加组件:**
  - Click 标记地点或标记朋友. For a video it opens the tag editor on the timeline; click 「+ 添加标记」.
  - In 选择标记 → 用户, pick **SGLang**:
    - The official account has the orange SGL avatar and userId `657172690000000020031d3f`.
    - It is usually first under 为你推荐. If it is not there, search 「SGLang」.
    - Look-alikes such as llllsglang or SGLang666 are wrong.
  - Confirm the inner dialog with 确定. A 「SGLang」 tag appears on the video at 00:00.
  - Confirm the outer dialog with 确定. The card now reads 「已添加1个标记」.
- **标记 gotchas:**
  - The user list item and the dialog buttons need real mouse clicks; JS `.click()` on `.sign-item` does nothing.
  - In a narrow pane the outer dialog's 确定 is off-screen. Click it through JS: the visible `button` whose text is
    确定.
  - A new marker gets a temporary timestamp id. An existing marker's `.marker-container` id is the userId.
- **Publish:** 发布 → `/publish/success`. Check `/new/note-manager`; new notes show 审核中 for a few minutes.

## Editing a published 小红书 note (e.g. adding the 标记 afterwards)

- **Open:** in `/new/note-manager`, find the card by `.note-card__title`. The edit button is the
  `.note-card__action-btn` just before `.note-card__action-btn--del`, and JS `.click()` works on it. This opens
  `/publish/update`.
- **Publish:** the 发布 button lives inside the `<xhs-publish-btn>` shadow DOM, so JS cannot reach it. Click it by
  coordinates, using the centre of `xhs-publish-btn`'s rect × (screenshot width / innerWidth). Leave the viewport
  unemulated here; an emulated size misaligned the coordinates.
- **Result:** the page shows 「更新成功」 at `/publish/editSuccess`.
- **Batch limit:** `browser_batch` takes at most 25 actions per call.
