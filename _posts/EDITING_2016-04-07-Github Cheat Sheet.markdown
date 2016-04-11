---
layout: post
title:  "Github Cheat Sheet"
date:   2016-04-07 10:00:00 -0500
categories: general
---

**Add existing project to repo**
[github][add-existing]

{% highlight bash %}
git init
git add .
git commit -m "First commit"
git remote add origin https://github.com/some/repo.git
git push origin master
{% endhighlight %}

**Force overwrite of remote files**
[stackoverflow][force-overwrite]

{% highlight bash %}
git push -f <remote> <branch>
git push -f origin master
{% endhighlight %}



**Start server to check markdown** [stackoverflow][grip]

{% highlight bash %}
grip README.md
{% endhighlight %}

[force-overwrite]: http://stackoverflow.com/questions/10510462/force-git-push-to-overwrite-remote-files
[add-existing]: https://help.github.com/articles/adding-an-existing-project-to-github-using-the-command-line/
[grip]: http://stackoverflow.com/questions/7694887/is-there-a-command-line-utility-for-rendering-github-flavored-markdown

