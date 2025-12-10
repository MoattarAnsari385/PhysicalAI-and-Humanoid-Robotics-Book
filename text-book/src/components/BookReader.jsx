import React, { useState, useContext } from 'react';
import { useParams } from 'react-router-dom';
import { ThemeContext } from './App';

const BookReader = () => {
  const { moduleId, chapterId } = useParams();
  const { theme, language, toggleLanguage } = useContext(ThemeContext);

  const [fontSize, setFontSize] = useState(16);
  const [themeMode, setThemeMode] = useState('dark'); // 'dark', 'light', 'sepia'
  const [bookmarks, setBookmarks] = useState([]);
  const [notes, setNotes] = useState({});
  const [showNotesPanel, setShowNotesPanel] = useState(false);

  const content = {
    english: {
      readingMode: "Reading Mode",
      fontSize: "Font Size",
      theme: "Theme",
      chapter: "Chapter",
      of: "of",
      next: "Next",
      previous: "Previous",
      bookmarks: "Bookmarks",
      addBookmark: "Add Bookmark",
      notes: "Notes",
      addNote: "Add Note",
      save: "Save",
      cancel: "Cancel",
      module1: "Module 1: Robotic Nervous System (ROS 2)",
      module2: "Module 2: The Digital Twin (Gazebo & Simulation)",
      module3: "Module 3: The AI-Robot Brain (NVIDIA Isaac)",
      module4: "Module 4: Vision-Language-Action (VLA)"
    },
    urdu: {
      readingMode: "پڑھنے کا موڈ",
      fontSize: "فونٹ سائز",
      theme: "تھیم",
      chapter: "چیپٹر",
      of: "کا",
      next: "اگلا",
      previous: "پچھلا",
      bookmarks: "صفحات کے نشان",
      addBookmark: "صفحہ کا نشان لگائیں",
      notes: "نوٹس",
      addNote: "نوٹ شامل کریں",
      save: "محفوظ کریں",
      cancel: "منسوخ کریں",
      module1: "ماڈیول 1: روبوٹک نروس سسٹم (ROS 2)",
      module2: "ماڈیول 2: ڈیجیٹل ٹوئن (گیزبو اور سیمولیشن)",
      module3: "ماڈیول 3: AI روبوٹ براہن (NVIDIA آئیساک)",
      module4: "ماڈیول 4: وژن لینگویج ایکشن (VLA)"
    }
  };

  const currentContent = content[language];

  // Mock chapter data
  const mockChapters = {
    'module-1-ros2': [
      { id: 'intro', title: language === 'english' ? 'Introduction to ROS 2' : 'ROS 2 کا تعارف' },
      { id: 'nodes-topics', title: language === 'english' ? 'Nodes, Topics, and Services' : 'نودس، ٹاپکس، اور سروسز' },
      { id: 'architecture', title: language === 'english' ? 'ROS 2 Architecture' : 'ROS 2 آرکیٹیکچر' }
    ],
    'module-2-simulation': [
      { id: 'intro', title: language === 'english' ? 'Introduction to Simulation' : 'سیمولیشن کا تعارف' },
      { id: 'gazebo', title: language === 'english' ? 'Gazebo Simulation' : 'گیزبو سیمولیشن' },
      { id: 'digital-twin', title: language === 'english' ? 'Digital Twin Concepts' : 'ڈیجیٹل ٹوئن کے تصورات' }
    ],
    'module-3-ai': [
      { id: 'intro', title: language === 'english' ? 'Introduction to AI Perception' : 'AI تاثر کا تعارف' },
      { id: 'isaac', title: language === 'english' ? 'NVIDIA Isaac Integration' : 'NVIDIA آئیساک انٹیگریشن' },
      { id: 'navigation', title: language === 'english' ? 'AI Navigation Systems' : 'AI نیویگیشن سسٹمز' }
    ],
    'module-4-vla': [
      { id: 'intro', title: language === 'english' ? 'Introduction to VLA' : 'VLA کا تعارف' },
      { id: 'voice-mapping', title: language === 'english' ? 'Voice Command Mapping' : 'وائس کمانڈ میپنگ' },
      { id: 'integration', title: language === 'english' ? 'VLA Integration' : 'VLA انٹیگریشن' }
    ]
  };

  const currentModuleChapters = mockChapters[moduleId] || [];
  const currentChapterIndex = currentModuleChapters.findIndex(ch => ch.id === chapterId);
  const currentChapter = currentModuleChapters[currentChapterIndex];

  const handleAddBookmark = () => {
    const bookmark = {
      id: Date.now(),
      chapter: currentChapter?.title,
      moduleId,
      timestamp: new Date().toISOString()
    };
    setBookmarks([...bookmarks, bookmark]);
  };

  const handleAddNote = (noteText) => {
    if (!noteText.trim()) return;

    const note = {
      id: Date.now(),
      text: noteText,
      timestamp: new Date().toISOString(),
      chapter: currentChapter?.title
    };

    setNotes({
      ...notes,
      [chapterId]: [...(notes[chapterId] || []), note]
    });
  };

  const getThemeStyles = () => {
    switch (themeMode) {
      case 'light':
        return { background: '#ffffff', color: '#000000' };
      case 'sepia':
        return { background: '#f4e4bc', color: '#5c4329' };
      default: // dark
        return { background: '#0c0c0c', color: '#ffffff' };
    }
  };

  const themeStyles = getThemeStyles();

  return (
    <div
      className="min-h-screen transition-colors duration-300"
      style={themeStyles}
    >
      {/* Header */}
      <header className="border-b border-white/20" style={{ borderColor: themeMode === 'light' ? '#e5e7eb' : 'rgba(255,255,255,0.2)' }}>
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1
                className="text-xl font-bold"
                style={{ fontFamily: language === 'english' ? "'Playfair Display', serif" : 'Noto Naskh Arabic, serif' }}
              >
                {mockModules.find(m => m.id === moduleId)?.title || moduleId}
              </h1>
              {currentChapter && (
                <p className="text-sm text-gray-500 mt-1">
                  {currentContent.chapter} {currentChapterIndex + 1} {currentContent.of} {currentModuleChapters.length}: {currentChapter.title}
                </p>
              )}
            </div>

            <div className="flex items-center gap-4">
              {/* Theme Toggle */}
              <select
                value={themeMode}
                onChange={(e) => setThemeMode(e.target.value)}
                className="bg-black/20 text-white border border-white/20 rounded-lg px-3 py-1 text-sm"
                style={{
                  background: themeMode === 'light' ? '#f3f4f6' : 'rgba(0,0,0,0.2)',
                  color: themeMode === 'light' ? '#000000' : '#ffffff',
                  border: themeMode === 'light' ? '1px solid #d1d5db' : '1px solid rgba(255,255,255,0.2)'
                }}
              >
                <option value="dark">{language === 'english' ? 'Dark' : 'سیاہ'}</option>
                <option value="light">{language === 'english' ? 'Light' : 'روشنی'}</option>
                <option value="sepia">{language === 'english' ? 'Sepia' : 'سیپیا'}</option>
              </select>

              {/* Font Size Control */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setFontSize(Math.max(12, fontSize - 2))}
                  className="px-2 py-1 bg-black/20 text-white rounded border border-white/20 text-sm hover:bg-white/20 transition-colors"
                  style={{
                    background: themeMode === 'light' ? '#f3f4f6' : 'rgba(0,0,0,0.2)',
                    color: themeMode === 'light' ? '#000000' : '#ffffff',
                    border: themeMode === 'light' ? '1px solid #d1d5db' : '1px solid rgba(255,255,255,0.2)'
                  }}
                >
                  A-
                </button>
                <span className="text-sm" style={{ color: themeMode === 'light' ? '#000000' : '#ffffff' }}>
                  {fontSize}px
                </span>
                <button
                  onClick={() => setFontSize(Math.min(24, fontSize + 2))}
                  className="px-2 py-1 bg-black/20 text-white rounded border border-white/20 text-sm hover:bg-white/20 transition-colors"
                  style={{
                    background: themeMode === 'light' ? '#f3f4f6' : 'rgba(0,0,0,0.2)',
                    color: themeMode === 'light' ? '#000000' : '#ffffff',
                    border: themeMode === 'light' ? '1px solid #d1d5db' : '1px solid rgba(255,255,255,0.2)'
                  }}
                >
                  A+
                </button>
              </div>

              {/* Language Toggle */}
              <button
                onClick={toggleLanguage}
                className="px-4 py-2 bg-black/20 text-white border border-white/20 rounded-lg text-sm hover:bg-white/20 transition-colors duration-300"
                style={{
                  background: themeMode === 'light' ? '#f3f4f6' : 'rgba(0,0,0,0.2)',
                  color: themeMode === 'light' ? '#000000' : '#ffffff',
                  border: themeMode === 'light' ? '1px solid #d1d5db' : '1px solid rgba(255,255,255,0.2)'
                }}
              >
                {language === 'english' ? 'اردو' : 'English'}
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Navigation Controls */}
        <div className="flex justify-between items-center mb-8">
          <button
            onClick={() => {
              if (currentChapterIndex > 0) {
                // Navigate to previous chapter
              }
            }}
            disabled={currentChapterIndex === 0}
            className="px-6 py-3 bg-gradient-to-r from-[#E00070] to-[#FF7A00] text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {currentContent.previous}
          </button>

          <div className="text-center">
            <h2
              className="text-2xl font-bold"
              style={{
                fontFamily: language === 'english' ? "'Playfair Display', serif" : 'Noto Naskh Arabic, serif',
                fontSize: `${fontSize * 1.2}px`
              }}
            >
              {currentChapter?.title}
            </h2>
          </div>

          <button
            onClick={() => {
              if (currentChapterIndex < currentModuleChapters.length - 1) {
                // Navigate to next chapter
              }
            }}
            disabled={currentChapterIndex === currentModuleChapters.length - 1}
            className="px-6 py-3 bg-gradient-to-r from-[#E00070] to-[#FF7A00] text-white rounded-lg hover:opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {currentContent.next}
          </button>
        </div>

        {/* Content Area */}
        <div
          className="prose prose-lg max-w-none transition-all duration-300"
          style={{
            fontSize: `${fontSize}px`,
            fontFamily: language === 'english' ? 'Poppins, sans-serif' : 'Noto Naskh Arabic, serif',
            lineHeight: language === 'urdu' ? '2.0' : '1.8',
            color: themeMode === 'light' ? '#000000' : '#ffffff'
          }}
        >
          {/* Mock content - in a real implementation, this would come from the actual textbook content */}
          <h1>{currentChapter?.title}</h1>

          <p className="lead">
            {language === 'english'
              ? 'This is a sample chapter from the Physical AI & Humanoid Robotics textbook. The content is fully responsive and adapts to the selected language, font size, and theme preferences.'
              : 'یہ فزیکل ای آئی اور ہیومنوڈ روبوٹکس ٹیکسٹ بک کا نمونہ چیپٹر ہے۔ مواد مکمل طور پر جواب دہ ہے اور منتخب کردہ زبان، فونٹ سائز اور تھیم کی ترجیحات کے مطابق ایڈجسٹ ہوجاتا ہے۔'}
          </p>

          <p>
            {language === 'english'
              ? 'The chapter content would include detailed explanations of robotics concepts, code examples, diagrams, and practical exercises tailored to humanoid robotics applications.'
              : 'چیپٹر کا مواد میں روبوٹکس کے تصورات کی تفصیلی وضاحتیں، کوڈ کے نمونے، ڈائریم اور ہیومنوڈ روبوٹکس ایپلی کیشنز کے مطابق عملی مشقیں شامل ہوں گی۔'}
          </p>

          <div className="mt-8 p-6 rounded-xl border"
               style={{
                 background: themeMode === 'light' ? '#f9fafb' : 'rgba(255,255,255,0.05)',
                 border: themeMode === 'light' ? '1px solid #e5e7eb' : '1px solid rgba(255,255,255,0.2)'
               }}>
            <h3 className="font-semibold mb-3"
                style={{ color: themeMode === 'light' ? '#000000' : '#ffffff' }}>
              {language === 'english' ? 'Key Concepts' : 'اہم تصورات'}
            </h3>
            <ul className="list-disc list-inside space-y-2">
              <li style={{ color: themeMode === 'light' ? '#374151' : '#d1d5db' }}>
                {language === 'english' ? 'Fundamental robotics principles' : 'روبوٹکس کے بنیادی اصول'}
              </li>
              <li style={{ color: themeMode === 'light' ? '#374151' : '#d1d5db' }}>
                {language === 'english' ? 'ROS 2 communication patterns' : 'ROS 2 کے مواصلاتی نمونے'}
              </li>
              <li style={{ color: themeMode === 'light' ? '#374151' : '#d1d5db' }}>
                {language === 'english' ? 'Simulation environments' : 'سیمولیشن ماحول'}
              </li>
              <li style={{ color: themeMode === 'light' ? '#374151' : '#d1d5db' }}>
                {language === 'english' ? 'AI perception and navigation' : 'AI تاثر اور نیویگیشن'}
              </li>
            </ul>
          </div>

          <p className="mt-6">
            {language === 'english'
              ? 'Each chapter builds upon previous concepts while introducing new robotics technologies and applications. Practical examples and exercises help reinforce learning through hands-on experience.'
              : 'ہر چیپٹر پچھلے تصورات پر تعمیر کرتا ہے جبکہ نئے روبوٹکس ٹیکنالوجیز اور ایپلی کیشنز کو متعارف کراتا ہے۔ عملی مثالیں اور مشقیں ہاتھوں ہاتھ تجربے کے ذریعے سیکھنے کو مضبوط بناتی ہیں۔'}
          </p>
        </div>
      </div>

      {/* Fixed Controls Bar */}
      <div className="fixed bottom-6 left-6 right-6 max-w-4xl mx-auto bg-black/20 backdrop-blur-sm rounded-2xl p-4 border border-white/20"
           style={{
             background: themeMode === 'light' ? 'rgba(249, 250, 251, 0.8)' : 'rgba(0, 0, 0, 0.2)',
             borderColor: themeMode === 'light' ? '#d1d5db' : 'rgba(255, 255, 255, 0.2)',
             backdropFilter: 'blur(20px)'
           }}>
        <div className="flex justify-between items-center">
          <div className="flex gap-2">
            <button
              onClick={handleAddBookmark}
              className="px-4 py-2 bg-white/10 text-white rounded-lg text-sm hover:bg-white/20 transition-colors border border-white/20"
              style={{
                background: themeMode === 'light' ? '#f3f4f6' : 'rgba(255,255,255,0.1)',
                color: themeMode === 'light' ? '#000000' : '#ffffff',
                border: themeMode === 'light' ? '1px solid #d1d5db' : '1px solid rgba(255,255,255,0.2)'
              }}
            >
              {currentContent.addBookmark}
            </button>

            <button
              onClick={() => setShowNotesPanel(!showNotesPanel)}
              className="px-4 py-2 bg-white/10 text-white rounded-lg text-sm hover:bg-white/20 transition-colors border border-white/20"
              style={{
                background: themeMode === 'light' ? '#f3f4f6' : 'rgba(255,255,255,0.1)',
                color: themeMode === 'light' ? '#000000' : '#ffffff',
                border: themeMode === 'light' ? '1px solid #d1d5db' : '1px solid rgba(255,255,255,0.2)'
              }}
            >
              {currentContent.notes}
            </button>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-white/80 text-sm" style={{ color: themeMode === 'light' ? '#6b7280' : '#ffffff' }}>
              {currentContent.fontSize}:
            </span>
            <input
              type="range"
              min="12"
              max="24"
              value={fontSize}
              onChange={(e) => setFontSize(parseInt(e.target.value))}
              className="w-24"
            />
            <span className="text-white/80 text-sm" style={{ color: themeMode === 'light' ? '#6b7280' : '#ffffff' }}>
              {fontSize}px
            </span>
          </div>
        </div>
      </div>

      {/* Notes Panel */}
      {showNotesPanel && (
        <div className="fixed bottom-24 left-6 right-6 max-w-4xl mx-auto bg-black/30 backdrop-blur-sm rounded-2xl p-4 border border-white/20 z-50"
             style={{
               background: themeMode === 'light' ? 'rgba(249, 250, 251, 0.9)' : 'rgba(0, 0, 0, 0.3)',
               borderColor: themeMode === 'light' ? '#d1d5db' : 'rgba(255, 255, 255, 0.2)',
               backdropFilter: 'blur(20px)'
             }}>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold" style={{ color: themeMode === 'light' ? '#000000' : '#ffffff' }}>
              {currentContent.notes}
            </h3>
            <button
              onClick={() => setShowNotesPanel(false)}
              className="text-white/60 hover:text-white text-xl"
              style={{ color: themeMode === 'light' ? '#6b7280' : '#ffffff' }}
            >
              ×
            </button>
          </div>

          <div className="space-y-4">
            {(notes[chapterId] || []).map(note => (
              <div
                key={note.id}
                className="p-3 rounded-lg border"
                style={{
                  background: themeMode === 'light' ? '#f9fafb' : 'rgba(255,255,255,0.05)',
                  border: themeMode === 'light' ? '1px solid #e5e7eb' : '1px solid rgba(255,255,255,0.2)',
                  color: themeMode === 'light' ? '#374151' : '#d1d5db'
                }}
              >
                {note.text}
                <small className="text-xs text-gray-500 block mt-1">
                  {new Date(note.timestamp).toLocaleString()}
                </small>
              </div>
            ))}

            <textarea
              placeholder={language === 'english' ? 'Add your note here...' : 'یہاں اپنا نوٹ شامل کریں...'}
              className="w-full p-3 rounded-lg border resize-none"
              rows="3"
              style={{
                background: themeMode === 'light' ? '#f9fafb' : 'rgba(255,255,255,0.05)',
                border: themeMode === 'light' ? '1px solid #e5e7eb' : '1px solid rgba(255,255,255,0.2)',
                color: themeMode === 'light' ? '#000000' : '#ffffff'
              }}
              onKeyPress={(e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                  handleAddNote(e.target.value);
                  e.target.value = '';
                }
              }}
            ></textarea>

            <div className="flex gap-2">
              <button
                onClick={(e) => {
                  const textarea = e.target.parentElement.previousElementSibling;
                  handleAddNote(textarea.value);
                  textarea.value = '';
                }}
                className="px-4 py-2 bg-gradient-to-r from-[#E00070] to-[#FF7A00] text-white rounded text-sm"
              >
                {currentContent.save}
              </button>
              <button
                onClick={() => setShowNotesPanel(false)}
                className="px-4 py-2 bg-white/10 text-white rounded text-sm border border-white/20 hover:bg-white/20 transition-colors"
                style={{
                  background: themeMode === 'light' ? '#f3f4f6' : 'rgba(255,255,255,0.1)',
                  color: themeMode === 'light' ? '#000000' : '#ffffff',
                  border: themeMode === 'light' ? '1px solid #d1d5db' : '1px solid rgba(255,255,255,0.2)'
                }}
              >
                {currentContent.cancel}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BookReader;