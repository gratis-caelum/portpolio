(function($) {
    'use strict';

    let blogPosts = [];
    let filteredPosts = [];
    let currentPage = 1;
    const postsPerPage = 6;
    let currentCategory = '';
    let currentSearchTerm = '';

    // Initialize blog functionality
    $(document).ready(function() {
        if (window.location.pathname.includes('blog.html')) {
            initializeBlogPage();
        } else if (window.location.pathname.includes('blog-post.html')) {
            initializeBlogPost();
        }
    });

    // Initialize blog listing page
    function initializeBlogPage() {
        loadBlogPosts();
        setupEventListeners();
    }

    // Initialize individual blog post page
    function initializeBlogPost() {
        const urlParams = new URLSearchParams(window.location.search);
        const postSlug = urlParams.get('post');
        if (postSlug) {
            loadBlogPost(postSlug);
        } else {
            showError('블로그 포스트를 찾을 수 없습니다.');
        }
    }

    // Load blog posts from JSON
    function loadBlogPosts() {
        $.getJSON('posts.json')
            .done(function(data) {
                blogPosts = data.posts || [];
                filteredPosts = [...blogPosts];
                renderBlogPosts();
                renderPagination();
                hideLoading();
            })
            .fail(function() {
                showNoPosts();
                hideLoading();
            });
    }

    // Load individual blog post
    function loadBlogPost(slug) {
        // First load the post metadata
        $.getJSON('posts.json')
            .done(function(data) {
                const post = data.posts.find(p => p.slug === slug);
                if (!post) {
                    showError('포스트를 찾을 수 없습니다.');
                    return;
                }
                
                // Update meta information
                updatePostMeta(post);
                
                // Load and render markdown content
                $.get(`posts/${post.slug}.md`)
                    .done(function(markdownContent) {
                        renderMarkdown(markdownContent, post);
                        setupPostNavigation(post, data.posts);
                    })
                    .fail(function() {
                        showError('포스트 내용을 불러올 수 없습니다.');
                    });
            })
            .fail(function() {
                showError('포스트 정보를 불러올 수 없습니다.');
            });
    }

    // Update post meta information
    function updatePostMeta(post) {
        document.title = `${post.title} | Byeongchan Kim`;
        $('#post-title-meta').attr('content', `${post.title} | Byeongchan Kim`);
        $('#post-description-meta').attr('content', post.excerpt);
        $('#post-keywords-meta').attr('content', post.tags.join(', '));
        
        $('#post-title').text(post.title);
        $('#post-date').text(formatDate(post.date));
        $('#post-category').text(post.category);
        $('#post-reading-time').text(`${post.readingTime || 5}분 읽기`);
        
        // Render tags
        const tagsHtml = post.tags.map(tag => 
            `<span class="tag">${tag}</span>`
        ).join('');
        $('#post-tags').html(tagsHtml);
    }

    // Render markdown content
    function renderMarkdown(markdownContent, post) {
        // Configure marked options
        marked.setOptions({
            highlight: function(code, lang) {
                if (Prism.languages[lang]) {
                    return Prism.highlight(code, Prism.languages[lang], lang);
                }
                return code;
            },
            breaks: true,
            gfm: true
        });

        const htmlContent = marked.parse(markdownContent);
        $('#post-content').html(htmlContent);
        
        // Re-highlight code blocks
        if (typeof Prism !== 'undefined') {
            Prism.highlightAll();
        }
    }

    // Setup post navigation (prev/next)
    function setupPostNavigation(currentPost, allPosts) {
        const currentIndex = allPosts.findIndex(p => p.slug === currentPost.slug);
        
        if (currentIndex > 0) {
            const prevPost = allPosts[currentIndex - 1];
            $('#prev-post').show().find('.nav-title')
                .text(prevPost.title)
                .attr('href', `blog-post.html?post=${prevPost.slug}`);
        }
        
        if (currentIndex < allPosts.length - 1) {
            const nextPost = allPosts[currentIndex + 1];
            $('#next-post').show().find('.nav-title')
                .text(nextPost.title)
                .attr('href', `blog-post.html?post=${nextPost.slug}`);
        }
    }

    // Setup event listeners for blog page
    function setupEventListeners() {
        // Search functionality
        $('#blog-search').on('input', function() {
            currentSearchTerm = $(this).val().toLowerCase();
            filterPosts();
        });

        // Category filter
        $('#category-filter').on('change', function() {
            currentCategory = $(this).val();
            filterPosts();
        });
    }

    // Filter posts based on search term and category
    function filterPosts() {
        filteredPosts = blogPosts.filter(post => {
            const matchesSearch = !currentSearchTerm || 
                post.title.toLowerCase().includes(currentSearchTerm) ||
                post.excerpt.toLowerCase().includes(currentSearchTerm) ||
                post.tags.some(tag => tag.toLowerCase().includes(currentSearchTerm));
            
            const matchesCategory = !currentCategory || post.category === currentCategory;
            
            return matchesSearch && matchesCategory;
        });

        currentPage = 1;
        renderBlogPosts();
        renderPagination();
    }

    // Render blog posts
    function renderBlogPosts() {
        const startIndex = (currentPage - 1) * postsPerPage;
        const endIndex = startIndex + postsPerPage;
        const postsToShow = filteredPosts.slice(startIndex, endIndex);

        if (postsToShow.length === 0) {
            showNoPosts();
            return;
        }

        const postsHtml = postsToShow.map(post => `
            <article class="blog-post-card">
                <div class="blog-post-image">
                    <a href="blog-post.html?post=${post.slug}">
                        <img src="${post.image || 'images/pic02.jpg'}" alt="${post.title}" />
                    </a>
                </div>
                <div class="blog-post-content">
                    <div class="post-meta">
                        <span class="post-date">${formatDate(post.date)}</span>
                        <span class="post-category">${post.category}</span>
                    </div>
                    <h3 class="major">
                        <a href="blog-post.html?post=${post.slug}">${post.title}</a>
                    </h3>
                    <p>${post.excerpt}</p>
                    <div class="post-tags">
                        ${post.tags.map(tag => `<span class="tag">${tag}</span>`).join('')}
                    </div>
                    <div class="post-actions">
                        <a href="blog-post.html?post=${post.slug}" class="special">읽기</a>
                    </div>
                </div>
            </article>
        `).join('');

        $('#blog-posts-grid').html(postsHtml);
    }

    // Render pagination
    function renderPagination() {
        const totalPages = Math.ceil(filteredPosts.length / postsPerPage);
        
        if (totalPages <= 1) {
            $('#blog-pagination').empty();
            return;
        }

        let paginationHtml = '';

        // Previous button
        if (currentPage > 1) {
            paginationHtml += `<li><a href="#" class="button small" data-page="${currentPage - 1}">이전</a></li>`;
        }

        // Page numbers
        const startPage = Math.max(1, currentPage - 2);
        const endPage = Math.min(totalPages, startPage + 4);

        for (let i = startPage; i <= endPage; i++) {
            const isActive = i === currentPage ? ' active' : '';
            paginationHtml += `<li><a href="#" class="page${isActive}" data-page="${i}">${i}</a></li>`;
        }

        // Next button
        if (currentPage < totalPages) {
            paginationHtml += `<li><a href="#" class="button small" data-page="${currentPage + 1}">다음</a></li>`;
        }

        $('#blog-pagination').html(paginationHtml);

        // Setup pagination event listeners
        $('#blog-pagination a').on('click', function(e) {
            e.preventDefault();
            const page = parseInt($(this).data('page'));
            if (page && page !== currentPage) {
                currentPage = page;
                renderBlogPosts();
                renderPagination();
                $('html, body').animate({ scrollTop: 0 }, 300);
            }
        });
    }

    // Utility functions
    function formatDate(dateString) {
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        return new Date(dateString).toLocaleDateString('ko-KR', options);
    }

    function hideLoading() {
        $('#blog-loading').hide();
    }

    function showNoPosts() {
        $('#blog-loading').hide();
        $('#blog-no-posts').show();
    }

    function showError(message) {
        $('#post-content').html(`
            <div class="error-message">
                <h3>오류</h3>
                <p>${message}</p>
                <a href="blog.html" class="button">블로그로 돌아가기</a>
            </div>
        `);
    }

})(jQuery);