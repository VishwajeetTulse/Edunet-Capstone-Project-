import React from 'react';
import { Gamepad2, GraduationCap, Trophy, User } from 'lucide-react';

interface Persona {
  id: string;
  name: string;
  icon: React.ElementType;
  description: string;
  data: any;
  color: string;
}

const PERSONAS: Persona[] = [
  {
    id: 'gamer',
    name: 'The Gamer',
    icon: Gamepad2,
    description: 'High screen time, irregular sleep, sedentary.',
    color: 'bg-purple-100 text-purple-600',
    data: {
      age: 16,
      sex: 1, // Male
      hours: 10, // Extreme
      height: 175,
      weight: 80,
      hr: 85,
      systolic: 130,
      diastolic: 85,
      sds: 60, // Poor sleep (Higher is worse)
      paq: 1.5, // Low activity
      cgas: 60, // Moderate functioning
    }
  },
  {
    id: 'student',
    name: 'The Student',
    icon: GraduationCap,
    description: 'High academic stress, moderate activity.',
    color: 'bg-blue-100 text-blue-600',
    data: {
      age: 15,
      sex: 1, // Male
      hours: 6,
      height: 160,
      weight: 55,
      hr: 75,
      systolic: 115,
      diastolic: 75,
      sds: 30, // Good sleep
      paq: 2.5, // Average activity
      cgas: 75, // Good functioning
    }
  },
  {
    id: 'athlete',
    name: 'The Athlete',
    icon: Trophy,
    description: 'Low screen time, physically active.',
    color: 'bg-emerald-100 text-emerald-600',
    data: {
      age: 17,
      sex: 1, // Male
      hours: 2,
      height: 180,
      weight: 75,
      hr: 60,
      systolic: 110,
      diastolic: 70,
      sds: 30, // Good sleep
      paq: 4.5, // High activity
      cgas: 90, // Excellent functioning
    }
  }
];

interface Props {
  onSelect: (data: any) => void;
  onCustom: () => void;
}

export default function PersonaSelector({ onSelect, onCustom }: Props) {
  return (
    <div className="max-w-4xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700 w-full">
      <div className="text-center space-y-4">
        <h1 className="text-4xl md:text-5xl font-black text-gray-900 tracking-tight">
          Who are we analyzing?
        </h1>
        <p className="text-xl text-gray-500">
          Select a persona to quick-start, or create a custom profile.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {PERSONAS.map((persona) => (
          <button
            key={persona.id}
            onClick={() => onSelect(persona.data)}
            className="group relative flex flex-col items-center p-8 bg-white rounded-3xl shadow-sm border-2 border-transparent hover:border-teal-500 hover:shadow-xl transition-all duration-300"
          >
            <div className={`p-4 rounded-2xl mb-6 ${persona.color} group-hover:scale-110 transition-transform`}>
              <persona.icon className="w-8 h-8" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">{persona.name}</h3>
            <p className="text-center text-gray-500 text-sm leading-relaxed">
              {persona.description}
            </p>
          </button>
        ))}
      </div>

      <div className="text-center pt-8">
        <button
          onClick={onCustom}
          className="inline-flex items-center px-8 py-4 bg-white text-gray-900 rounded-full font-bold shadow-sm hover:shadow-md border border-gray-200 hover:bg-gray-50 transition-all"
        >
          <User className="w-5 h-5 mr-2 text-gray-400" />
          Start from Scratch
        </button>
      </div>
    </div>
  );
}