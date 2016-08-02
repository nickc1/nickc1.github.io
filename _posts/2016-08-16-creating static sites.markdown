---
layout: post
title:  "Cheap, Quick, Fast, Static Sites"
date:   2016-01-30 08:00:00 -0500
categories: jekyll
---

"GIVE ME A VIDEO!!" Okay here you go : [MicrowaveSam - Youtube](vid)

I unfortunately found this video late in the game. You can follow this great youtube video to set up your site. There are only minor tweaks below.

-----

It took me longer than I care to admit to figure out how to host a static site cheaply and without doing much configuration. Following this should get you up and running with your own domain in very little time given you already have a static site up and running.

For creating your site, I suggest [jekyll](jekyll), which this site is built with, or grabbing a HTML template from [HTML5 UP](html5up). There are tons of tutorials out there on creating a static site, so I am not going to go into those here. This tutorial is going to assume that you have a site that you want to host somewhere for extremely cheap.

First I googled around and I found a [post][blog-post] which lists cheap hosting options. This got me off and moving on my journey.

I decided to go with Amazon S3 as I figured this would be a good entry into the Amazon ecosystem. It also scales really well, so if this site ever becomes popular (yea, okay) it can handle the traffic.

So first things first. I bought a domain name from [namecheap.com](namecheap). After I had my domain ([nickcortale.me](me)), I went over to amazon web services to set up my s3 bucket.

So first thing to do is create an account for [Amazon web services](aws) if you don't have one already. You should be able to get it free for the first year if you're a student which is a second plus! Next we need to click on s3 which is on the far right under services (I'm sure the location of the logo will change in the future).

Next you want to create a bucket and call it `www.yourdomain.com`, but replace `yourdomin.com` with whatever you purchased. This is extremely important. After your bucket is created, it should look like the image below.

![amazon s3](/assets/static_site/s3_bucket.png){: .center-image }

Next we need to allow our bucket to host websites. You will find this by clicking `properties` and then going down to `staic website hosting`. Fill in the fields like below. This assumes your landing page is named `index.html`.

![amazon s3](/assets/static_site/enable_hosting.png){: .center-image }

Next we want to upload all our stuff to the bucket. After it has been uploaded we need to set all the files to public so that when people access the site, they can see the files.

![amazon s3](/assets/static_site/make_public.png){: .center-image }

Okay. Now make a note of the website address that amazon assigns you. This can be found under the static site tab. We will need this when we register on namecheap.

![amazon s3](/assets/static_site/enable_hosting.png){: .center-image }

Register a domain name. Remember it must be the same domain name as what you called your bucket. So if you called your bucket `www.eatvegetables.com` you must buy the domain `eatvegetables.com`.

Next is the part that tripped me up so badly when I was trying to figure this out. We need to point our domain to the our s3 bucket. Amazon gives some instructions, but it turns out all you have to do is edit your DNS records. It should look like the image below.

![amazon s3](/assets/static_site/dns_settings.png){: .center-image }

That should be it. Your website should be up and running.


[blog-post]: http://alignedleft.com/resources/cheap-web-hosting
[vid]: https://youtu.be/KTFWnnVi1oA
[jekyll]: https://jekyllrb.com/
[html5up]: https://html5up.net/
[namecheap]: https://www.namecheap.com/
[aws]: https://aws.amazon.com/
[me]: http://www.nickcortale.me/
