import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { ThemeContext } from './App';

const OnboardingFlow = () => {
  const { theme, language, toggleLanguage } = useContext(ThemeContext);
  const navigate = useNavigate();
  const [currentSlide, setCurrentSlide] = useState(0);

  const content = {
    english: {
      skip: "Skip",
      next: "Next",
      getStarted: "Get Started",
      slide1Title: "Welcome to Physical AI & Robotics",
      slide1Subtitle: "Your journey to mastering humanoid robotics begins here",
      slide2Title: "Comprehensive Learning Path",
      slide2Subtitle: "From ROS 2 fundamentals to advanced VLA integration",
      slide3Title: "Start Your Journey",
      slide3Subtitle: "Ready to become a robotics expert?",
      modules: "4 Comprehensive Modules",
      support: "24/7 Support",
      certificate: "Certificate of Completion"
    },
    urdu: {
      skip: "چھوڑ دیں",
      next: "اگلا",
      getStarted: "شروع کریں",
      slide1Title: "فزیکل ای آئی اور روبوٹکس میں خوش آمدید",
      slide1Subtitle: "ہیومنوڈ روبوٹکس کے ماسٹر بننے کا آپ کا سفر یہاں سے شروع ہوتا ہے",
      slide2Title: "جامع سیکھنے کا راستہ",
      slide2Subtitle: "ROS 2 کے فنڈامینلز سے لے کر ایڈوانس VLA انٹیگریشن تک",
      slide3Title: "اپنا سفر شروع کریں",
      slide3Subtitle: "کیا آپ روبوٹکس کے ماہر بننے کے لیے تیار ہیں؟",
      modules: "4 جامع ماڈیولز",
      support: "24/7 سپورٹ",
      certificate: "مکمل ہونے کا سرٹیفکیٹ"
    }
  };

  const currentContent = content[language];

  const slides = [
    {
      title: currentContent.slide1Title,
      subtitle: currentContent.slide1Subtitle,
      icon: (
        <svg className="w-24 h-24 mx-auto mb-8 text-[#E00070]" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
        </svg>
      ),
      features: [
        { text: currentContent.modules, icon: '📚' },
        { text: currentContent.support, icon: '👤' },
        { text: currentContent.certificate, icon: '📜' }
      ]
    },
    {
      title: currentContent.slide2Title,
      subtitle: currentContent.slide2Subtitle,
      icon: (
        <svg className="w-24 h-24 mx-auto mb-8 text-[#FF7A00]" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
      ),
      features: [
        { text: language === 'english' ? 'Module 1: ROS 2' : 'ماڈیول 1: ROS 2', icon: '⚙️' },
        { text: language === 'english' ? 'Module 2: Simulation' : 'ماڈیول 2: سیمولیشن', icon: '🎮' },
        { text: language === 'english' ? 'Module 3: AI Perception' : 'ماڈیول 3: AI تاثر', icon: '🧠' },
        { text: language === 'english' ? 'Module 4: VLA Integration' : 'ماڈیول 4: VLA انٹیگریشن', icon: '🤖' }
      ]
    },
    {
      title: currentContent.slide3Title,
      subtitle: currentContent.slide3Subtitle,
      icon: (
        <svg className="w-24 h-24 mx-auto mb-8 text-[#E00070]" fill="currentColor" viewBox="0 0 24 24">
          <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
        </svg>
      ),
      features: [
        { text: language === 'english' ? 'Hands-on Projects' : 'ہاتھوں ہاتھ منصوبے', icon: '🛠️' },
        { text: language === 'english' ? 'Real Robotics' : 'حقیقی روبوٹکس', icon: '🤖' },
        { text: language === 'english' ? 'Industry Ready' : 'صنعت کے لیے تیار', icon: '🏭' }
      ]
    }
  ];

  const handleNext = () => {
    if (currentSlide < slides.length - 1) {
      setCurrentSlide(currentSlide + 1);
    } else {
      // Complete onboarding
      localStorage.setItem('onboardingComplete', 'true');
      navigate('/');
    }
  };

  const handleSkip = () => {
    localStorage.setItem('onboardingComplete', 'true');
    navigate('/');
  };

  const handleDotClick = (index) => {
    setCurrentSlide(index);
  };

  return (
    <div
      className="min-h-screen flex items-center justify-center p-4"
      style={{ background: theme.primaryGradient }}
    >
      <div className="w-full max-w-md mx-auto">
        {/* Skip Button */}
        <div className="flex justify-end mb-4">
          <button
            onClick={handleSkip}
            className="text-white/70 hover:text-white transition-colors duration-300 text-sm font-medium"
          >
            {currentContent.skip}
          </button>
        </div>

        {/* Slide Content */}
        <div className="bg-white/10 backdrop-blur-lg rounded-3xl p-8 border border-white/20 shadow-2xl">
          <div className="text-center mb-8">
            {slides[currentSlide].icon}

            <h2
              className="text-2xl font-bold text-white mb-4 leading-tight"
              style={{
                fontFamily: language === 'english' ? "'Playfair Display', serif" : 'Noto Naskh Arabic, serif'
              }}
            >
              {slides[currentSlide].title}
            </h2>

            <p className="text-white/80 mb-8 leading-relaxed">
              {slides[currentSlide].subtitle}
            </p>

            {/* Features List */}
            <div className="space-y-3 mb-8">
              {slides[currentSlide].features.map((feature, index) => (
                <div
                  key={index}
                  className="flex items-center justify-center p-3 bg-black/20 rounded-xl border border-white/10"
                >
                  <span className="text-xl mr-3">{feature.icon}</span>
                  <span className="text-white font-medium">{feature.text}</span>
                </div>
              ))}
            </div>

            {/* Progress Dots */}
            <div className="flex justify-center items-center gap-3 mb-8">
              {slides.map((_, index) => (
                <button
                  key={index}
                  onClick={() => handleDotClick(index)}
                  className={`w-3 h-3 rounded-full transition-all duration-300 ${
                    index === currentSlide
                      ? 'bg-[#E00070] w-6'
                      : 'bg-white/30 hover:bg-white/50'
                  }`}
                />
              ))}
            </div>

            {/* Action Button */}
            <button
              onClick={handleNext}
              className="w-full py-4 bg-gradient-to-r from-[#E00070] to-[#FF7A00] text-white rounded-xl font-bold text-lg hover:shadow-lg transition-all duration-300 transform hover:scale-105"
            >
              {currentSlide === slides.length - 1 ? currentContent.getStarted : currentContent.next}
            </button>
          </div>
        </div>

        {/* Language Toggle */}
        <div className="text-center mt-6">
          <button
            onClick={toggleLanguage}
            className="px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full text-white border border-white/20 hover:bg-white/20 transition-all duration-300 text-sm"
          >
            {language === 'english' ? 'اردو' : 'English'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default OnboardingFlow;