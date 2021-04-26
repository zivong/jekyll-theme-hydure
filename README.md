# Hydure <!-- omit in toc -->

Hydure is a concise two-column blog theme for jekyll. It is built on the [Pure.css](https://github.com/pure-css/pure) framework.

## Highlight Features <!-- omit in toc -->

- [Open Color](https://github.com/yeun/open-color)
- Dark mode, via [`prefers-color-scheme`](https://developer.mozilla.org/en-US/docs/Web/CSS/@media/prefers-color-scheme)

## Table of Contents <!-- omit in toc -->

- [Installation](#installation)
- [Usage](#usage)
  - [Global Configuration](#global-configuration)
  - [Homepage](#homepage)
  - [Custom Head](#custom-head)
- [Contributing](#contributing)
- [Development](#development)
- [License](#license)

## Installation

Add this line to your Jekyll site's `Gemfile`:

```ruby
gem "jekyll-theme-hydure"
```

And add this line to your Jekyll site's `_config.yml`:

```yaml
theme: jekyll-theme-hydure
```

And then execute:

```shell
bundle
```

Or install it yourself as:

```shell
gem install jekyll-theme-hydure
```

If your website is hosted on GitHub Pages, you can install this theme via [`jekyll-remote-theme`](https://github.com/benbalter/jekyll-remote-theme).

Add the following to your `Gemfile`:

```ruby
gem "jekyll-remote-theme"
```

And add this line to your Jekyll site's `_config.yml`:

```yml
plugins:
  - jekyll-remote-theme
```

Add the following to your site's `_config.yml`:

```yml
remote_theme: zivong/jekyll-theme-hydure
```

## Usage

### Global Configuration

| Variable | Type | Default | Specification |
| -------- | ---- | ------- | ------------- |
| `lang` | String | `en` | The language of pages; The value can be overwritten by the `lang` variable on each page. |
| `title` | String | --- | The title of the website |
| `tagline` | String | --- | The tagline of the website |
| `cover` | String | --- | The URL of the sidebar cover image; The value can be overwritten by the `cover` variable on each page. |

### Homepage

You can create a homepage for your site by setting `layout: home` in your `index.html`.

### Custom Head

Hydure leaves a placeholder to allow defining custom head. All you need to do is putting data into `_includes/custom-head.html`, and they would be automatically included in `<head>`.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/zivong/jekyll-theme-hydure. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## Development

To set up your environment to develop this theme, run `bundle install`.

Your theme is setup just like a normal Jekyll site! To test your theme, run `bundle exec jekyll serve` and open your browser at `http://localhost:4000`. This starts a Jekyll server using your theme. Add pages, documents, data, etc. like normal to test your theme's contents. As you make modifications to your theme and to your content, your site will regenerate and you should see the changes in the browser after a refresh, just like normal.

When your theme is released, only the files in `_layouts`, `_includes`, `_sass` and `assets` tracked with Git will be bundled.
To add a custom directory to your theme-gem, please edit the regexp in `jekyll-theme-hydure.gemspec` accordingly.

## License

The theme is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).
