const reveal = new IntersectionObserver((entries) => {
  entries.forEach((entry) => { if (entry.isIntersecting) entry.target.classList.add('visible'); });
}, { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach((el) => reveal.observe(el));

const scenarios = [
  { health: 82, vibration: '3.8', temperature: 67, rul: 18 },
  { health: 64, vibration: '5.1', temperature: 74, rul: 9 },
  { health: 91, vibration: '2.2', temperature: 58, rul: 31 }
];
let scenario = 0;
document.getElementById('simulateBtn').addEventListener('click', () => {
  scenario = (scenario + 1) % scenarios.length;
  const s = scenarios[scenario];
  document.getElementById('healthScore').textContent = s.health;
  document.getElementById('vibration').textContent = s.vibration;
  document.getElementById('temperature').textContent = s.temperature;
  document.getElementById('rul').textContent = s.rul;
  document.querySelector('.health-ring').style.setProperty('--value', s.health);
});
