---
layout: post
title: What's Jekyll?
category: [Example]
tags: [jekyll]
---

[Jekyll](https://jekyllrb.com) is a static site generator, an open-source tool for creating simple yet powerful websites of all shapes and sizes. 

## To create a Github Page with a Jekylle theme
Assume you have a github account: `https://github.com/yourname`
### Fork Theme
1. Go to [Jekyll-Theme-Hydure](https://github.com/zivong/jekyll-theme-hydure) click [Fork] <br />
2. Click [Settings], **Repository name** enter YourBlogName (Ex. `AI-course`), click [Rename]<br />
3. Delete 2 old branches, keep `master` branch <br />
4. Create new Branch `gh-pages` <br />
5. Set default branch to `gh-pages`
### Add Theme
6. Edit Gemfile to add: <br />
   `theme  "jekyll-theme-hydure"`
7. Edit _config.yml to add: <br />
   `plugin:` <br />
   &emsp;`- jekyll-remote-theme` <br />
   `remote_theme: zivong/jekyll-theme-hydure`
### Change Cover.jpg
8. Upload a cover.jpg (1920x2880) to **_layouts** <br /> 
9. Modify **_layouts/default.hmtl** to change path of cover.jpg <br />
   `assign cover = site.cover | default: "cover.jpg"`
### Change Title & Tagline
10. Modify **_config.yml**  <br />
    &emsp;`- title: AI course` <br />
    &emsp;`- tagline: Deep Learning Course` <br />
11. Modify **_data/navigation.yml** to change url<br />
    `- tile Download` <br />
    &emsp;`url: https://github.com/rkuo2000/AI-course/archive/refs/heads/master.zip` <br />
    `- title: Github` <br />
    &emsp;`url: https://github.com/rkuo2000/AI-course`
### Edit Posts
12. Modify **_posts/*.md**, upload or remove .md) <br />
    (posts display latest dated .md on the top) <br />
    `2021-11-15-Lesson1.md` <br />
    `2021-03-01-whats-jekyll.md` <br />
### Open Github Page    
    Click [https://yourname.github.io/YourBlogName](https://rkuo2000.github.io/AI-course)

