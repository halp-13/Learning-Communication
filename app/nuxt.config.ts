// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({

  ssr: false,
  // modules: ["usebootstrap",],
  app: {
   
    head: {
      title: 'Nuxt.js TypeScript',
      meta: [
        { charset: 'utf-8' },
        { name: 'viewport', content: 'width=device-width, initial-scale=1' },
      ],
      link: [
        { rel: 'icon', type: 'image/x-icon', href: '/favicon.ico' },
        { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css?family=Roboto:300,400,500,700|Material+Icons' },
        { rel: 'stylesheet', href: '/css/bootstrap.min.css' },
        { rel: 'stylesheet', href: '/css/style.css' },


      ],

    },
  },
  devtools: { enabled: false },

  compatibilityDate: '2025-02-11',
})