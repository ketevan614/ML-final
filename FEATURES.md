

## Calendar Features

- `Year`, `Month`, `Quarter`, `WeekOfYear`, `DayOfYear`  
  გაყიდვებში არის სეზონურობა, ამიტომ მოდელს უნდა ჰქონდეს ინფორმაცია წლის რომელ პერიოდში ხდება პროგნოზი.

- `WeekSin`, `WeekCos`, `MonthSin`, `MonthCos`  
  კვირისა და თვის ციკლურობას უკეთ აჩვენებს. მაგალითად, 52-ე კვირა და 1-ლი კვირა ერთმანეთთან ახლოსაა, მაგრამ უბრალო რიცხვებით ეს კარგად არ ჩანს.

## Holiday Features

- `is_super_bowl`, `is_labor_day`, `is_thanksgiving`, `is_christmas`  
  `IsHoliday` მხოლოდ იმას გვეუბნება, რომ კვირა holiday-ა. ცალკე holiday feature-ები საჭიროა, რადგან Thanksgiving-ს და Christmas-ს გაყიდვებზე განსხვავებული ეფექტი აქვთ.

- `is_pre_christmas`, `is_december`, `DaysToChristmas`  
  Christmas-ის ეფექტი ხშირად თვითონ holiday კვირამდე იწყება, ამიტომ დავამატეთ pre-Christmas პერიოდის feature-ები.

## Markdown Features

- `MarkDown1`-`MarkDown5` missing მნიშვნელობები შევავსეთ `0`-ით.  
  ეს საჭიროა, რადგან მოდელების უმეტესობა missing მნიშვნელობებზე პირდაპირ ვერ მუშაობს.

- `MarkDown1_missing`-`MarkDown5_missing`, `MarkdownMissingCount`  
  markdown მონაცემები ბევრ ადრეულ თარიღში საერთოდ არ იყო ხელმისაწვდომი. missing flag-ები მოდელს აძლევს ინფორმაციას, რომ მნიშვნელობა რეალურად არ იყო ცნობილი.

- `TotalMarkdown`, `TotalMarkdown_log1p`, `AnyMarkdown`  
  ეს აჩვენებს მთლიან promotional/discount აქტივობას კონკრეტულ კვირაში.

## Temperature and Fuel Features

- `Temperature`, `Fuel_Price`  
  raw ცვლადები დავტოვეთ, რადგან ამინდი და საწვავის ფასი შეიძლება მოქმედებდეს მაღაზიაში ვიზიტებსა და გაყიდვებზე.

- `TempCold`, `TempHot`, `TempMild`  
  აჩვენებს ამინდის მარტივ კატეგორიებს: ძალიან ცივი, ძალიან ცხელი ან საშუალო.

- `TempComfortDistance`  
  აჩვენებს რამდენად შორს არის ტემპერატურა კომფორტული დიაპაზონიდან.

- `TemperatureStoreDeviation`, `FuelPriceStoreDeviation`  
  აჩვენებს კონკრეტულ კვირაში ტემპერატურა ან საწვავის ფასი რამდენად განსხვავდება ამ მაღაზიის ჩვეულებრივი მნიშვნელობისგან.

- `TempFuelInteraction`, `ComfortDistanceFuelInteraction`  
  ეხმარება მოდელს ისწავლოს კომბინირებული ეფექტი, მაგალითად ცუდი ამინდი და მაღალი საწვავის ფასი ერთად.

## Economic Features

- `CPI`, `Unemployment`  
  ეკონომიკური მდგომარეობა შეიძლება მოქმედებდეს მომხმარებლის ხარჯვის ქცევაზე.

- `CPI_missing`, `Unemployment_missing`  
  test პერიოდში ზოგი ეკონომიკური მნიშვნელობა missing იყო. მნიშვნელობები შევავსეთ, მაგრამ flag-ებით შევინარჩუნეთ ინფორმაცია, რომ ისინი თავდაპირველად არ იყო ცნობილი.

## Sales History Features

- `sales_lag_52`  
  იგივე Store-Dept-ის გაყიდვა ერთი წლით ადრე. ეს ძლიერი feature-ია, რადგან Walmart-ის გაყიდვებში yearly seasonality ჩანს.

- `same_week_history_mean`  
  იგივე Store-Dept-ის იმავე კვირის საშუალო გაყიდვა წინა წლებში. ეს არის თქვენი იდეის safe ვერსია.

- `store_dept_history_mean`, `dept_week_history_mean`, `store_week_history_mean`  
  fallback historical averages sparse series-ებისთვის, როცა კონკრეტულ Store-Dept-ს საკმარისი ისტორია არ აქვს.

- `*_missing` flags sales-history feature-ებისთვის  
  აჩვენებს, როდის გამოვიყენეთ fallback, რადგან წინა წლის ან ისტორიული ინფორმაცია არ არსებობდა.

## Important Note

ყველა sales-history feature გაკეთებულია ისე, რომ არ გამოიყენოს მომავალი გაყიდვები. ეს აუცილებელია, რადგან წინააღმდეგ შემთხვევაში გვექნებოდა target leakage და validation შედეგი არ იქნებოდა სანდო.
