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
- **Typing the 简介:** focus `.ql-editor`, then `document.execCommand('insertText', false, line)` and
  `execCommand('insertParagraph')` for each line. The counter under the box (`N/2000`) must then show the
  draft's length. Never use `quill.setText(body)`: it changes only the view, so 7 videos went live with an
  empty 简介.
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
  - Before each topic, put the caret at the end of the last paragraph. After Enter it can sit before the topic
    just inserted, and the next one lands in front of it.
  - Count topics inside `.tiptap.ProseMirror` only; `document.querySelectorAll('a.tiptap-topic')` returns
    several nodes per topic.
  - Always include #人工智能 #大模型 #SGLang. `#量化` is the finance topic; use `#模型量化`.
- **内容类型声明:** 添加内容类型声明 → 笔记含AI合成内容.
- **封面 (always upload it):**
  - 小红书 makes the cover from the video's first frame. Our cover frame fades its text in, so that frame is
    often a bare orange or photo background.
  - Always upload `NN_<slug>_封面_B站.png` (4:3): 编辑封面 → 上传封面图片 → 完成. The `.cover-edit-entry` has
    zero size until the thumbnail is hovered; a JS `.click()` on it opens the editor all the same.
  - Then look at the image actually used (§Checks).
- **标记 SGLang (required), in 添加组件:**
  - Click 标记地点或标记朋友. For a video it opens the tag editor on the timeline; click 「+ 添加标记」.
  - In 选择标记 → 用户, pick **SGLang**:
    - The official account has the orange SGL avatar and userId `657172690000000020031d3f`.
    - It is usually first under 为你推荐. If it is not there, search 「SGLang」.
    - Look-alikes such as llllsglang or SGLang666 are wrong, and the search sometimes lists one of them
      first. Pick the row whose avatar URL contains `1kjclhpkia005pbhe9ko679vunm0ne8`.
    - The 标记 entry stays disabled until the video upload has finished.
  - Confirm the inner dialog with 确定. A 「SGLang」 tag appears on the video at 00:00.
  - Confirm the outer dialog with 确定. The card now reads 「已添加1个标记」.
- **标记 gotchas:**
  - The user list item and the dialog buttons need real mouse clicks; JS `.click()` on `.sign-item` does nothing.
  - In a narrow pane the outer dialog's 确定 is off-screen. Click it through JS: the visible `button` whose text is
    确定.
  - A new marker gets a temporary timestamp id. An existing marker's `.marker-container` id is the userId.
- **Publish:** 发布 → `/publish/success`. Check `/new/note-manager`; new notes show 审核中 for a few minutes.
  发布 sits in a closed shadow root, so it needs a real click. In a narrow pane, clear the viewport emulation and
  click 收起侧边栏 first, so the button bar fits on screen.

## Checks (every publish, on both sites)

Both sites drop input without saying so: a 简介 that never reached the form, a tag that vanished, a cover made
from a blank frame, the previous video still selected in the file dialog, a 标记 lost when the video was
swapped. So a publish is not done when the button is clicked. It is done when every check below has passed
against what the site actually stored. Report each one to the user; anything that failed or was skipped is
said plainly.

**1. Before opening the upload pages,** write the B站 简介 and the 小红书 body to files and run:

```bash
$SEV_PY -I $SEV_SKILL/scripts/publish_check.py --video ~/Desktop/小红书上传/NN_<slug>.mp4 \
    --render $P/renders/video.mp4 --cover ~/Desktop/小红书上传/NN_<slug>_封面_B站.png \
    --bili-title "<B站标题>" --xhs-title "<小红书标题>" --bili-desc desc.txt --xhs-body body.txt
```

Fix every ✗ first. Note the duration and both text lengths it prints; the later checks compare against them.

**2. B站, before 立即投稿:** run this in the page and compare each field with the draft.

```js
const t = document.body.innerText;
({ title: document.querySelector('input[placeholder*="标题"]').value,
   uploaded: t.match(/[^\n]*上传完成/g),                         // file name + 上传完成
   desc: t.match(/(\d+)\s*\/\s*2000/)?.[1],                     // == draft length, never 0
   declare: document.querySelector('input[placeholder="请选择符合您视频内容的创作声明"]')?.value,
   zone: t.match(/分区\n\n([^\n]*)/)?.[1],                       // 人工智能
   tags: t.match(/标签\n\n([\s\S]*?)\n\n还可以添加/)?.[1].split('\n\n'),  // the drafted tags, no auto tags
   collection: t.match(/合集\n([^\n]*)\n清除/)?.[1] })
```

The uploaded file name must be this video's (`NN_<slug>`), and the auto-filled title before you replaced it
must have been that name too. The cover dialog's two `x/vu/web/cover/up` responses must both be 200.

**3. B站, after 稿件投递成功:** read what was stored.

```js
const a = (await fetch('https://member.bilibili.com/x/web/archives?status=is_pubing,pubed,not_pubed&ps=3',
  {credentials: 'include'}).then(r => r.json())).data.arc_audits[0].Archive;
const x = (await fetch('https://member.bilibili.com/x/vupre/web/archive/view?bvid=' + a.bvid,
  {credentials: 'include'}).then(r => r.json())).data;
({ bvid: a.bvid, title: x.archive.title, state: x.archive.state_desc, desc: x.archive.desc.length,
   tags: x.archive.tag, zone: x.archive.human_type2?.name, parts: x.videos.map(v => v.duration),
   cover: x.archive.cover })
```

- `desc` equals the draft length.
- `tags` are the drafted ones. B站 silently drops some tags (RTX4090, 高性能计算); re-add or replace them.
- `zone` is 人工智能.
- `parts` holds one entry within 1 s of the local duration.
- Download `cover` and look at it.
- If the video went into a 合集: `x2/creative/web/season/section?id=<section id>` lists this BV last.
  Shorten its 单集标题 in the 合集 editor, then read the section again to confirm the new title stuck.

**4. 小红书, before 发布:**

```js
const t = document.body.innerText, ed = document.querySelector('.tiptap.ProseMirror');
({ title: document.querySelector('input[placeholder="填写标题会有更多赞哦"]').value,
   file: t.match(/NN_<slug>\.mp4/)?.[0], hd: t.includes('检测为高清视频'),
   topics: [...ed.querySelectorAll('a.tiptap-topic')].map(a => a.innerText),
   body: ed.innerText.length, ai: t.includes('笔记含AI合成内容'),
   mark: t.match(/已添加\d+个标记|标记地点或标记朋友/)?.[0],          // SGLang video: 已添加1个标记
   cover: [...document.querySelectorAll('[style*="ros-preview"]')].map(e => e.style.backgroundImage.slice(5, -2))[0] })
```

- The topics are exactly the drafted ones, at the end of the body.
- The SGLang row was picked by its avatar id (§标记).
- Download `cover` and look at it. It must be the 4:3 cover, not a blank frame.

**5. 小红书, after 发布:**

- The URL reaches `/publish/success`, or `/publish/publish?...&published=true`.
- In `/new/note-manager`, `全部 N` went up by one.
- The first card shows the title and a duration that matches the local file (小红书 rounds down, so 760.7 s
  shows 12:40), with 审核中.

## Editing a published 小红书 note (e.g. adding the 标记 afterwards)

- **Open:** in `/new/note-manager`, find the card by `.note-card__title`. The edit button is the
  `.note-card__action-btn` just before `.note-card__action-btn--del`, and JS `.click()` works on it. This opens
  `/publish/update`.
- **Publish:** the 发布 button lives inside the `<xhs-publish-btn>` shadow DOM, so JS cannot reach it. Click it by
  coordinates, using the centre of `xhs-publish-btn`'s rect × (screenshot width / innerWidth). Leave the viewport
  unemulated here; an emulated size misaligned the coordinates.
- **Result:** the page shows 「更新成功」 at `/publish/editSuccess`.
- **Swapping the video (重新上传) drops the 标记.** Re-add it in the same edit, then reopen `/publish/update` and
  check that it shows 「已添加1个标记」.
- **Batch limit:** `browser_batch` takes at most 25 actions per call.
