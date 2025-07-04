/**
 * Language Switcher Module
 * 다국어 지원을 위한 언어 전환 모듈
 * 이벤트 위임을 사용하여 효율적인 언어 전환 기능 제공
 */

(function() {
    'use strict';

    // 기본 언어 설정
    const DEFAULT_LANGUAGE = 'ko';
    
    // 언어 전환 초기화
    function initLanguageSwitcher() {
        // 페이지 로드 시 저장된 언어 적용
        const savedLanguage = localStorage.getItem('language') || DEFAULT_LANGUAGE;
        applyTranslations(savedLanguage);
        updateActiveButton(savedLanguage);
        
        // 이벤트 위임을 사용한 언어 버튼 클릭 이벤트
        document.addEventListener('click', function(event) {
            const button = event.target.closest('button[data-lang]');
            if (button) {
                const lang = button.getAttribute('data-lang');
                changeLanguage(lang);
            }
        });
    }
    
    /**
     * 언어를 변경하고 로컬 스토리지에 저장
     * @param {string} lang - 변경할 언어 코드 (ko, ja, en)
     */
    function changeLanguage(lang) {
        if (!isValidLanguage(lang)) {
            console.warn(`지원되지 않는 언어입니다: ${lang}`);
            return;
        }
        
        localStorage.setItem('language', lang);
        applyTranslations(lang);
        updateActiveButton(lang);
        
        // 언어 변경 이벤트 발생 (필요시 다른 모듈에서 감지 가능)
        document.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: lang } }));
    }
    
    /**
     * 페이지의 텍스트를 선택된 언어로 변경
     * @param {string} lang - 적용할 언어 코드
     */
    function applyTranslations(lang) {
        // translations 객체가 존재하는지 확인
        if (typeof translations === 'undefined') {
            console.error('translations 객체를 찾을 수 없습니다. language.js 파일이 올바르게 로드되었는지 확인하세요.');
            return;
        }
        
        const translation = translations[lang];
        if (!translation) {
            console.warn(`'${lang}' 언어에 대한 번역 데이터를 찾을 수 없습니다.`);
            return;
        }
        
        // 성능 최적화를 위해 DocumentFragment 사용
        const updates = [];
        
        for (const key in translation) {
            const element = document.getElementById(key);
            if (element) {
                updates.push({
                    element: element,
                    text: translation[key]
                });
            }
        }
        
        // 배치 업데이트로 리플로우 최소화
        updates.forEach(update => {
            update.element.textContent = update.text;
        });
    }
    
    /**
     * 활성 언어 버튼 스타일 업데이트
     * @param {string} lang - 현재 활성 언어
     */
    function updateActiveButton(lang) {
        const buttons = document.querySelectorAll('button[data-lang]');
        buttons.forEach(button => {
            if (button.getAttribute('data-lang') === lang) {
                button.classList.add('active');
            } else {
                button.classList.remove('active');
            }
        });
    }
    
    /**
     * 유효한 언어인지 확인
     * @param {string} lang - 확인할 언어 코드
     * @returns {boolean} 유효한 언어인지 여부
     */
    function isValidLanguage(lang) {
        return typeof translations !== 'undefined' && translations.hasOwnProperty(lang);
    }
    
    /**
     * 현재 활성 언어 반환
     * @returns {string} 현재 활성 언어 코드
     */
    function getCurrentLanguage() {
        return localStorage.getItem('language') || DEFAULT_LANGUAGE;
    }
    
    // DOM 로드 완료 시 초기화
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initLanguageSwitcher);
    } else {
        initLanguageSwitcher();
    }
    
    // 전역 함수로 노출 (필요시 외부에서 사용 가능)
    window.languageSwitcher = {
        changeLanguage: changeLanguage,
        getCurrentLanguage: getCurrentLanguage,
        applyTranslations: applyTranslations
    };
    
})(); 