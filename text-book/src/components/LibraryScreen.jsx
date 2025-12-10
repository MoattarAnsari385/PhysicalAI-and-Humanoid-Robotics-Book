import React, { useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { ThemeContext } from './App';

const LibraryScreen = () => {
  const { theme, language, toggleLanguage, user } = useContext(ThemeContext);
  const navigate = useNavigate();

  const content = {
    english: {
      library: "Library",
      myBooks: "My Books",
      continueReading: "Continue Reading",
      allModules: "All Modules",
      searchPlaceholder: "Search textbooks...",
      sortBy: "Sort by",
      newest: "Newest",
      progress: "Progress",
      completed: "Completed",
      inProgress: "In Progress",
      book1Title: "Physical AI & Humanoid Robotics",
      book1Subtitle: "University-level textbook for modern robotics",
      chapter: "Chapter",
      of: "of",
      lastAccessed: "Last accessed",
      timeRemaining: "time remaining"
    },
    urdu: {
      library: "لائبریری",
      myBooks: "میری کتب",
      continueReading: "پڑھنا جاری رکھیں",
      allBooks: "تمام کتب",
      searchPlaceholder: "کتب تلاش کریں...",
      sortBy: "چھانٹیں بلحاظ",
      newest: "تازہ ترین",
      progress: "پیشرفت",
      completed: "مکمل",
      inProgress: "جاری",
      book1Title: "فزیکل ای آئی اور ہیومنوڈ روبوٹکس",
      book1Subtitle: "جدید روبوٹکس کے لیے یونیورسٹی سطح کی ٹیکسٹ بک",
      chapter: "چیپٹر",
      of: "کا",
      lastAccessed: "آخری رسائی",
      timeRemaining: "باقی وقت"
    }
  };

  const currentContent = content[language];

  // Mock data for the library
  const mockLibraryData = {
    continueReading: [
      {
        id: 1,
        title: language === 'english' ? 'Module 3: AI Perception' : 'ماڈیول 3: AI تاثر',
        currentChapter: 4,
        totalChapters: 6,
        lastAccessed: '2024-01-15',
        progress: 67,
        thumbnail: 'module3-thumb.png'
      }
    ],
    allModules: [
      {
        id: 1,
        title: language === 'english' ? 'Module 1: Robotic Nervous System (ROS 2)' : 'ماڈیول 1: روبوٹک نروس سسٹم (ROS 2)',
        description: language === 'english' ? 'Fundamentals of ROS 2 communication and robotic systems' : 'ROS 2 کے مواصلات اور روبوٹک سسٹمز کے فنڈامینلز',
        chapters: 6,
        progress: 100,
        status: 'completed',
        lastAccessed: '2024-01-10',
        thumbnail: 'module1-thumb.png'
      },
      {
        id: 2,
        title: language === 'english' ? 'Module 2: The Digital Twin (Gazebo & Simulation)' : 'ماڈیول 2: ڈیجیٹل ٹوئن (گیزبو اور سیمولیشن)',
        description: language === 'english' ? 'Simulation environments and digital twin concepts' : 'سیمولیشن ماحول اور ڈیجیٹل ٹوئن کے تصورات',
        chapters: 5,
        progress: 100,
        status: 'completed',
        lastAccessed: '2024-01-12',
        thumbnail: 'module2-thumb.png'
      },
      {
        id: 3,
        title: language === 'english' ? 'Module 3: The AI-Robot Brain (NVIDIA Isaac)' : 'ماڈیول 3: AI روبوٹ براہن (NVIDIA آئیساک)',
        description: language === 'english' ? 'Perception, navigation, and AI integration with NVIDIA Isaac' : 'NVIDIA آئیساک کے ساتھ تاثر، نیویگیشن، اور AI انٹیگریشن',
        chapters: 6,
        progress: 67,
        status: 'in-progress',
        lastAccessed: '2024-01-15',
        thumbnail: 'module3-thumb.png'
      },
      {
        id: 4,
        title: language === 'english' ? 'Module 4: Vision-Language-Action (VLA)' : 'ماڈیول 4: وژن لینگویج ایکشن (VLA)',
        description: language === 'english' ? 'Integrating vision, language, and action for humanoid robots' : 'ہیومنوڈ روبوٹس کے لیے وژن، لینگویج، اور ایکشن کو مربوط کرنا',
        chapters: 6,
        progress: 20,
        status: 'in-progress',
        lastAccessed: '2024-01-16',
        thumbnail: 'module4-thumb.png'
      }
    ]
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString(language === 'english' ? 'en-US' : 'ur-PK', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const getStatusBadge = (status) => {
    const statusConfig = {
      completed: {
        text: language === 'english' ? 'Completed' : 'مکمل',
        color: 'bg-green-600/20 text-green-300 border-green-500/30'
      },
      'in-progress': {
        text: language === 'english' ? 'In Progress' : 'زیر عمل',
        color: 'bg-yellow-600/20 text-yellow-300 border-yellow-500/30'
      },
      locked: {
        text: language === 'english' ? 'Locked' : ' مقفل',
        color: 'bg-gray-600/20 text-gray-300 border-gray-500/30'
      }
    };

    const config = statusConfig[status] || statusConfig.locked;

    return (
      <span className={`px-3 py-1 rounded-full text-xs font-medium border ${config.color}`}>
        {config.text}
      </span>
    );
  };

  return (
    <div
      className="min-h-screen p-4"
      style={{
        background: theme.primaryGradient,
        fontFamily: language === 'english' ? 'Poppins, sans-serif' : 'Noto Naskh Arabic, serif'
      }}
    >
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-8 gap-4">
          <div>
            <h1
              className="text-3xl font-bold text-white mb-2"
              style={{ fontFamily: language === 'english' ? "'Playfair Display', serif" : 'Noto Naskh Arabic, serif' }}
            >
              {currentContent.library}
            </h1>
            <p className="text-white/70">
              {currentContent.myBooks} • {mockLibraryData.allModules.length} {language === 'english' ? 'modules' : 'ماڈیولز'}
            </p>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative">
              <input
                type="text"
                placeholder={currentContent.searchPlaceholder}
                className="w-64 px-4 py-2 bg-black/30 border border-white/20 rounded-xl text-white placeholder-white/50 focus:outline-none focus:border-[#E00070] transition-colors duration-300"
              />
              <svg className="absolute right-3 top-2.5 w-5 h-5 text-white/50" fill="currentColor" viewBox="0 0 24 24">
                <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
              </svg>
            </div>

            <select className="px-4 py-2 bg-black/30 border border-white/20 rounded-xl text-white focus:outline-none focus:border-[#E00070] transition-colors duration-300">
              <option>{currentContent.sortBy}</option>
              <option>{currentContent.newest}</option>
              <option>{currentContent.progress}</option>
            </select>
          </div>
        </div>

        {/* Continue Reading Section */}
        {mockLibraryData.continueReading.length > 0 && (
          <div className="mb-12">
            <h2 className="text-2xl font-bold text-white mb-6" style={{ fontFamily: language === 'english' ? "'Playfair Display', serif" : 'Noto Naskh Arabic, serif' }}>
              {currentContent.continueReading}
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {mockLibraryData.continueReading.map((book) => (
                <div
                  key={book.id}
                  className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:bg-white/20 transition-all duration-300 cursor-pointer group"
                  onClick={() => navigate(`/reader/module-${book.id}/chapter-${book.currentChapter}`)}
                >
                  <div className="flex items-start justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white group-hover:text-[#FF7A00] transition-colors">
                      {book.title}
                    </h3>
                    <span className="text-white/50 text-sm">{book.progress}%</span>
                  </div>

                  <div className="mb-4">
                    <div className="flex justify-between text-sm text-white/70 mb-2">
                      <span>
                        {currentContent.chapter} {book.currentChapter} {currentContent.of} {book.totalChapters}
                      </span>
                      <span>{formatDate(book.lastAccessed)}</span>
                    </div>
                    <div className="w-full bg-black/30 rounded-full h-2">
                      <div
                        className="bg-gradient-to-r from-[#E00070] to-[#FF7A00] h-2 rounded-full transition-all duration-300"
                        style={{ width: `${book.progress}%` }}
                      ></div>
                    </div>
                  </div>

                  <button className="w-full py-3 bg-gradient-to-r from-[#E00070] to-[#FF7A00] text-white rounded-xl font-medium hover:opacity-90 transition-opacity">
                    {language === 'english' ? 'Continue Reading' : 'پڑھنا جاری رکھیں'}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* All Modules Section */}
        <div>
          <h2 className="text-2xl font-bold text-white mb-6" style={{ fontFamily: language === 'english' ? "'Playfair Display', serif" : 'Noto Naskh Arabic, serif' }}>
            {currentContent.allModules}
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {mockLibraryData.allModules.map((module) => (
              <div
                key={module.id}
                className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:bg-white/20 transition-all duration-300 cursor-pointer group"
                onClick={() => navigate(`/docs/module-${module.id}-${module.id === 1 ? 'ros2' : module.id === 2 ? 'simulation' : module.id === 3 ? 'ai' : 'vla'}/intro`)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-white group-hover:text-[#FF7A00] transition-colors mb-2">
                      {module.title}
                    </h3>
                    <p className="text-white/70 text-sm mb-3">
                      {module.description}
                    </p>

                    <div className="flex items-center gap-4 text-sm text-white/60 mb-4">
                      <span>{module.chapters} {language === 'english' ? 'chapters' : 'ابواب'}</span>
                      <span>{language === 'english' ? 'Last accessed' : 'آخری رسائی'} {formatDate(module.lastAccessed)}</span>
                    </div>
                  </div>

                  <div className="ml-4">
                    {getStatusBadge(module.status)}
                  </div>
                </div>

                <div className="mb-4">
                  <div className="flex justify-between text-sm text-white/70 mb-2">
                    <span>{language === 'english' ? 'Progress' : 'پیشرفت'}</span>
                    <span>{module.progress}%</span>
                  </div>
                  <div className="w-full bg-black/30 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-[#E00070] to-[#FF7A00] h-2 rounded-full transition-all duration-300"
                      style={{ width: `${module.progress}%` }}
                    ></div>
                  </div>
                </div>

                <div className="flex gap-3">
                  <button className="flex-1 py-2 bg-white/10 text-white rounded-lg text-sm hover:bg-white/20 transition-colors border border-white/20">
                    {language === 'english' ? 'Details' : 'تفصیلات'}
                  </button>
                  <button className="flex-1 py-2 bg-gradient-to-r from-[#E00070] to-[#FF7A00] text-white rounded-lg text-sm hover:opacity-90 transition-opacity">
                    {module.status === 'completed'
                      ? (language === 'english' ? 'Review' : 'جائزہ')
                      : (language === 'english' ? 'Continue' : 'جاری رکھیں')
                    }
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Empty State (if no books) */}
        {mockLibraryData.allModules.length === 0 && (
          <div className="text-center py-16">
            <div className="w-24 h-24 mx-auto bg-white/10 rounded-full flex items-center justify-center mb-6">
              <svg className="w-12 h-12 text-white/50" fill="currentColor" viewBox="0 0 24 24">
                <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
              </svg>
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">
              {language === 'english' ? 'No modules yet' : 'ابھی تک کوئی ماڈیول نہیں'}
            </h3>
            <p className="text-white/70 mb-6">
              {language === 'english' ? 'Start exploring the robotics textbook to begin your learning journey' : 'اپنی سیکھنے کی رفتار شروع کرنے کے لیے روبوٹکس ٹیکسٹ بک دریافت کریں'}
            </p>
            <button
              onClick={() => navigate('/')}
              className="px-6 py-3 bg-gradient-to-r from-[#E00070] to-[#FF7A00] text-white rounded-xl font-medium hover:opacity-90 transition-opacity"
            >
              {language === 'english' ? 'Browse Modules' : 'ماڈیولز دیکھیں'}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default LibraryScreen;