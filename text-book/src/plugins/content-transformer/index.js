// Docusaurus plugin for content transformation (translation and personalization)
const path = require('path');

module.exports = function contentTransformerPlugin(context, options) {
  return {
    name: 'content-transformer-plugin',

    // Extend the default MDX loader to process content
    configureWebpack(config, isServer, utils) {
      return {
        module: {
          rules: [
            {
              test: /\.mdx?$/,
              use: [
                {
                  loader: require.resolve('@docusaurus/mdx-loader'),
                  options: {
                    remarkPlugins: [
                      // Add remark plugin to process personalization tags
                      () => (tree) => {
                        // This would process MDX AST to handle personalization
                        // For now, we'll focus on a simpler approach
                      }
                    ],
                    rehypePlugins: [],
                    staticDirs: path.resolve(__dirname, '../../static'),
                    ...options,
                  },
                },
              ],
            },
          ],
        },
      };
    },

    // Modify the loaded content
    async contentLoaded({ content, actions }) {
      // This function would transform the content based on user preferences
    },

    // Inject scripts/styles for dynamic content transformation
    injectHtmlTags() {
      return {
        postBodyTags: [
          `<script>
            // Client-side content transformation based on user preferences
            (function() {
              function updateContentBasedOnPreferences() {
                // Get user preferences from localStorage or context
                const preferences = JSON.parse(localStorage.getItem('userPreferences') || '{"level":"intermediate", "language":"en"}');

                // Apply language transformation
                if (preferences.language === 'ur') {
                  // Apply RTL layout
                  document.documentElement.dir = 'rtl';

                  // Apply Urdu-specific font
                  document.documentElement.style.fontFamily = '"Jameel Noori Nastaleeq", "Noto Nastaliq Urdu", "Urdu Typesetting", serif';

                  // Transform content to Urdu where available
                  const elements = document.querySelectorAll('*');
                  elements.forEach(el => {
                    if (el.children.length === 0 && el.textContent.trim()) { // Leaf elements
                      const originalText = el.textContent;
                      // This would use the translation API from our utils
                      // For now, we'll just apply styling changes
                      el.setAttribute('data-original-text', originalText);
                    }
                  });
                } else {
                  // Reset to LTR for English
                  document.documentElement.dir = 'ltr';
                  document.documentElement.style.fontFamily = '';
                }

                // Apply level-based content filtering
                if (preferences.level) {
                  // Show/hide content based on user level
                  document.querySelectorAll('[data-level]').forEach(el => {
                    if (el.getAttribute('data-level') === preferences.level ||
                        el.getAttribute('data-level') === 'all') {
                      el.style.display = 'block';
                    } else {
                      el.style.display = 'none';
                    }
                  });
                }
              }

              // Listen for preference changes
              window.addEventListener('userPreferencesChanged', function(e) {
                updateContentBasedOnPreferences();
              });

              // Run on initial load
              if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', updateContentBasedOnPreferences);
              } else {
                updateContentBasedOnPreferences();
              }
            })();
          </script>`
        ],
      };
    },
  };
};